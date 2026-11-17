"""Offline audio-boundary checks using small temporary WAV triples."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from realtime_speech_separation.evaluation.audio import AudioTriple, load_audio_triple
from realtime_speech_separation.evaluation.types import RecordingError


def write_triple(tmp_path: Path, separation_channels: int = 2) -> tuple[Path, Path, Path]:
    """Write an aligned float WAV triple without any enrollment file."""
    target = np.linspace(-1.5, 1.5, 128, dtype=np.float32)
    residual = np.linspace(0.5, -0.5, 128, dtype=np.float32)
    reference = np.stack((target, residual), axis=1)
    paths = (tmp_path / "call.wav", tmp_path / "call_reference.wav", tmp_path / "call_separated_hash.wav")
    for path, samples in zip(paths, (target + residual, reference, reference[:, :separation_channels]), strict=True):
        sf.write(path, samples, 16000, subtype="FLOAT")
    return paths


@pytest.mark.parametrize("separation_shape", ["mono", "single_channel", "stereo"])
def test_load_shapes_and_preserve_samples(tmp_path: Path, separation_shape: str) -> None:
    """Channel order, offset, amplitude, and length survive 16 kHz loading."""
    paths = write_triple(tmp_path)
    separated = np.arange(256, dtype=np.float32).reshape(128, 2) / 32 + 2
    if separation_shape == "mono":
        separated = separated[:, 0]
    elif separation_shape == "single_channel":
        separated = separated[:, :1]
    sf.write(paths[2], separated, 16000, subtype="FLOAT")

    audio = load_audio_triple(*paths)

    assert isinstance(audio, AudioTriple)
    assert audio.mixture.samples.shape == (128,)
    assert audio.reference.samples.shape == (2, 128)
    assert audio.separated.samples.shape == (2 if separation_shape == "stereo" else 1, 128)
    assert audio.sample_count == 128
    for path, loaded in zip(paths, (audio.mixture, audio.reference, audio.separated), strict=True):
        expected, _ = sf.read(path, dtype="float32", always_2d=True)
        np.testing.assert_array_equal(loaded.samples.reshape(expected.T.shape), expected.T)
        assert loaded.samples.dtype == np.float32
        assert loaded.native_sample_rate == 16000
        assert loaded.native_channels == expected.shape[1]
    assert not (tmp_path / "call_enrollment.wav").exists()


def test_stereo_mixture_is_downmixed(tmp_path: Path) -> None:
    """Preserve the CLI's mono-loading behavior for a stereo mixture."""
    paths = write_triple(tmp_path)
    mixture = np.tile(np.array([2, 4], dtype=np.float32), (128, 1))
    sf.write(paths[0], mixture, 16000, subtype="FLOAT")

    audio = load_audio_triple(*paths)

    assert isinstance(audio, AudioTriple)
    assert audio.mixture.native_channels == 2
    np.testing.assert_array_equal(audio.mixture.samples, np.full(128, 3, dtype=np.float32))


def test_resample_different_native_rates(tmp_path: Path) -> None:
    """Equal durations yield equal lengths after independent resampling."""
    paths = write_triple(tmp_path)
    rates = (8000, 48000, 16000)
    lengths = (801, 4806, 1602)
    for path, rate, length, channels in zip(paths, rates, lengths, (1, 2, 1), strict=True):
        samples = np.full((length, channels), 0.25, dtype=np.float32)
        sf.write(path, samples, rate, subtype="FLOAT")

    audio = load_audio_triple(*paths)

    assert isinstance(audio, AudioTriple)
    assert audio.sample_count == 1602
    for loaded, rate in zip((audio.mixture, audio.reference, audio.separated), rates, strict=True):
        assert loaded.native_sample_rate == rate
        assert loaded.samples.shape[-1] == 1602
        assert loaded.samples.dtype == np.float32
        np.testing.assert_allclose(loaded.samples[..., 100:-100], 0.25, atol=2e-4)


def test_pcm_subtype_is_accepted(tmp_path: Path) -> None:
    """Evaluation does not enforce floating-point WAV storage."""
    paths = write_triple(tmp_path)
    sf.write(paths[0], np.zeros(128, dtype=np.int16), 16000, subtype="PCM_16")
    audio = load_audio_triple(*paths)
    assert isinstance(audio, AudioTriple)
    assert audio.mixture.samples.dtype == np.float32


@pytest.mark.parametrize("file_index", [0, 1, 2])
@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_file_errors_retain_path_and_decoder_reason(tmp_path: Path, file_index: int, failure: str) -> None:
    """Decoder file errors become one recording error for the offending path."""
    paths = write_triple(tmp_path)
    path = paths[file_index]
    if failure == "missing":
        path.unlink()
    else:
        path.write_text("invalid WAV data")
    with pytest.raises((OSError, sf.LibsndfileError)) as decoder_error:
        sf.read(path, dtype="float32")

    error = load_audio_triple(*paths)

    assert error == RecordingError(str(path), str(decoder_error.value))


@pytest.mark.parametrize("file_index", [0, 1, 2])
def test_empty_waveform(tmp_path: Path, file_index: int) -> None:
    """Reject empty files on every side of the triple."""
    paths = write_triple(tmp_path)
    sf.write(paths[file_index], np.empty((0, 2 if file_index == 1 else 1)), 16000, subtype="FLOAT")
    error = load_audio_triple(*paths)
    assert isinstance(error, RecordingError)
    assert error.path == str(paths[file_index])
    assert "empty" in error.reason


@pytest.mark.parametrize("file_index", [0, 1, 2])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_nonfinite_waveform(tmp_path: Path, file_index: int, nonfinite: float, sample_rate: int) -> None:
    """Nonfinite input is rejected during loading or waveform validation."""
    paths = write_triple(tmp_path)
    samples = np.zeros((128, 2 if file_index == 1 else 1), dtype=np.float32)
    samples[50, -1] = nonfinite
    sf.write(paths[file_index], samples, sample_rate, subtype="FLOAT")
    error = load_audio_triple(*paths)
    assert isinstance(error, RecordingError)
    assert error.path == str(paths[file_index])
    assert "finite" in error.reason


@pytest.mark.parametrize(("file_index", "channels"), [(0, 3), (1, 1), (1, 3), (2, 3)])
def test_wrong_channel_count(tmp_path: Path, file_index: int, channels: int) -> None:
    """Reject mono references and unsupported multichannel recordings."""
    paths = write_triple(tmp_path)
    sf.write(paths[file_index], np.zeros((128, channels), dtype=np.float32), 16000, subtype="FLOAT")
    error = load_audio_triple(*paths)
    assert isinstance(error, RecordingError)
    assert error.path == str(paths[file_index])
    assert "channels" in error.reason


@pytest.mark.parametrize("file_index", [1, 2])
@pytest.mark.parametrize(("sample_rate", "length"), [(16000, 127), (16000, 129), (8000, 65)])
def test_unequal_resampled_lengths(tmp_path: Path, file_index: int, sample_rate: int, length: int) -> None:
    """Length mismatches are errors even when native sample rates differ."""
    paths = write_triple(tmp_path)
    channels = 2 if file_index == 1 else 1
    sf.write(paths[file_index], np.zeros((length, channels), dtype=np.float32), sample_rate, subtype="FLOAT")
    error = load_audio_triple(*paths)
    assert isinstance(error, RecordingError)
    assert error.path == str(paths[file_index])
    assert "Resampled length" in error.reason
    assert "mixture length 128" in error.reason


def test_silent_channels_are_valid(tmp_path: Path) -> None:
    """Silence is a metric eligibility concern rather than invalid audio."""
    paths = write_triple(tmp_path)
    for path, channels in zip(paths, (1, 2, 1), strict=True):
        sf.write(path, np.zeros((128, channels), dtype=np.float32), 16000, subtype="FLOAT")
    assert isinstance(load_audio_triple(*paths), AudioTriple)
