"""Load aligned evaluation triples without enrollment or separator resources."""

from dataclasses import dataclass
from os import PathLike
from typing import Literal

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from .types import RecordingError

EVALUATION_SAMPLE_RATE = 16000
MIN_RECORDING_SECONDS = 1


@dataclass(frozen=True)
class LoadedAudio:
    """Float32 audio at 16 kHz with its original file metadata.

    Samples have shape (n,) for the mixture, (2, n) for the reference, and
    (1, n) or (2, n) for the separation. Channels always retain T/R order.
    """

    samples: NDArray[np.float32]
    native_sample_rate: int
    native_channels: int


@dataclass(frozen=True)
class AudioTriple:
    """Validated mixture, reference, and separation with equal sample counts.

    Temporal alignment is the caller's responsibility; loading performs no
    delay removal, length correction, amplitude normalization, or clipping.
    """

    mixture: LoadedAudio
    reference: LoadedAudio
    separated: LoadedAudio

    @property
    def sample_count(self) -> int:
        """Return the shared recording length in samples at 16 kHz."""
        return self.mixture.samples.shape[-1]


def _load_audio(path: str, role: Literal["mixture", "reference", "separated"]) -> LoadedAudio | RecordingError:
    """Inspect native metadata, then load and validate audio at 16 kHz."""
    # Check native channels before mono loading can conceal an unsupported layout.
    try:
        metadata = sf.info(path)
    except (OSError, sf.LibsndfileError) as exc:
        return RecordingError(path, str(exc))
    expected_channels = (2,) if role == "reference" else (1, 2)
    if metadata.channels not in expected_channels:
        return RecordingError(path, f"{role} requires {expected_channels} channels; got {metadata.channels}")
    if metadata.frames == 0:
        return RecordingError(path, "Audio waveform is empty")

    # Let librosa handle float32 decoding, mixture downmixing, and resampling.
    try:
        samples, _ = librosa.load(path, sr=EVALUATION_SAMPLE_RATE, mono=role == "mixture", dtype=np.float32)
    except (OSError, sf.LibsndfileError, librosa.util.exceptions.ParameterError) as exc:
        return RecordingError(path, str(exc))
    if samples.ndim not in (1, 2):
        return RecordingError(path, f"Expected 1D or 2D audio; got shape {samples.shape}")
    if samples.shape[-1] == 0:
        return RecordingError(path, "Audio waveform is empty")
    if samples.shape[-1] < MIN_RECORDING_SECONDS * EVALUATION_SAMPLE_RATE:
        return RecordingError(path, f"Audio must be at least {MIN_RECORDING_SECONDS} second long after resampling")
    if not np.isfinite(samples).all():
        return RecordingError(path, "Audio waveform contains nonfinite samples")

    # Normalize target-only outputs to (1, n), preserving T/R channel order.
    if role == "separated" and samples.ndim == 1:
        samples = samples[np.newaxis, :]
    return LoadedAudio(samples, metadata.samplerate, metadata.channels)


def load_audio_triple(
    mixture_file: str | PathLike[str],
    reference_file: str | PathLike[str],
    separated_file: str | PathLike[str],
) -> AudioTriple | RecordingError:
    """Load three files as float32 at 16 kHz, returning the first input error.

    Mono or stereo mixtures are downmixed to (n,). References must have two
    channels; separations may have one or two. Each waveform must run for at least
    one second after resampling. Nonfinite waveforms and unequal resampled
    lengths are rejected without shifting, truncating, or padding. File errors
    retain the decoder's reason and the offending path.
    Unexpected programming errors propagate to the caller.
    """
    loaded = []
    # Validate at the loading boundary before any metric or VAD backend runs.
    for path, role in ((mixture_file, "mixture"), (reference_file, "reference"), (separated_file, "separated")):
        audio = _load_audio(str(path), role)
        if isinstance(audio, RecordingError):
            return audio
        if loaded and audio.samples.shape[-1] != loaded[0].samples.shape[-1]:
            return RecordingError(
                str(path),
                f"Resampled length {audio.samples.shape[-1]} does not match mixture length "
                f"{loaded[0].samples.shape[-1]} at {EVALUATION_SAMPLE_RATE} Hz",
            )
        loaded.append(audio)
    return AudioTriple(*loaded)
