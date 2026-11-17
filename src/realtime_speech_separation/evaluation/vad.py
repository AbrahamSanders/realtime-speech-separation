"""Offline Silero VAD and speech timing preparation for the evaluation pipeline.

Audio comes from the validated 16 kHz float32 loader. Reference preparation is
explicitly separate so callers can reuse it across one mixture's separations.
Only constructing SileroVad imports the optional VAD runtime.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .audio import EVALUATION_SAMPLE_RATE
from .types import ActivityCounts, ActivityResult, Channel, ChannelResult, Interval, SpeechRegion

MERGE_GAP_SAMPLES = int(1.0 * EVALUATION_SAMPLE_RATE)
UTTERANCE_MIN_ACTIVE_SAMPLES = int(0.5 * EVALUATION_SAMPLE_RATE)
OVERLAP_MIN_ACTIVE_SAMPLES = int(0.2 * EVALUATION_SAMPLE_RATE)
OVERLAP_MIN_SPAN_SAMPLES = int(0.5 * EVALUATION_SAMPLE_RATE)
ACTIVITY_FRAME_SAMPLES = int(0.1 * EVALUATION_SAMPLE_RATE)
_SILERO_PARAMETERS = {
    "sampling_rate": EVALUATION_SAMPLE_RATE,
    "threshold": 0.5,
    "neg_threshold": 0.35,
    "min_speech_duration_ms": 0,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 0,
    "return_seconds": False,
}


class VadBackendError(RuntimeError):
    """VAD execution failed; no speech population exists."""


class SpeechDetector(Protocol):
    """Injectable full-channel detector returning sorted, disjoint raw intervals."""

    def __call__(self, samples: NDArray[np.float32]) -> tuple[Interval, ...]:
        """Detect speech in a validated mono waveform at 16 kHz."""
        ...


class SileroVad:
    """One reusable CPU Silero JIT model, with independent state on every call.

    Construction requires the eval extra and reads the installed package's model
    without downloading assets. Silero's timestamp function handles state reset
    and gradient disabling. Instances are intended for sequential use. Runtime
    failures raise VadBackendError; unexpected programming errors propagate.
    Versions, model digest, and explicit settings are exposed for the pipeline's
    configuration identity. This adapter does not configure runtime threads.
    """

    def __init__(self) -> None:
        """Load and fingerprint the packaged model once, keeping imports lazy."""
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        self.backend_version = version("silero-vad")
        self.torch_version = torch.__version__
        model_path = files("silero_vad.data").joinpath("silero_vad.jit")
        model_bytes = model_path.read_bytes()
        self.model_digest = sha256(model_bytes).hexdigest()
        self._torch = torch
        self._timestamps = get_speech_timestamps
        self._model = load_silero_vad(onnx=False).cpu().eval()

    @property
    def parameters(self) -> dict[str, object]:
        """Return resolved explicit timestamp and execution settings for provenance."""
        return {**_SILERO_PARAMETERS, "model": "silero_vad.jit", "device": "cpu"}

    def __call__(self, samples: NDArray[np.float32]) -> tuple[Interval, ...]:
        """Use Silero's full-channel timestamps directly as raw sample intervals."""
        contiguous_samples = np.ascontiguousarray(samples)
        waveform = self._torch.from_numpy(contiguous_samples)
        try:
            timestamps = self._timestamps(waveform, self._model, **_SILERO_PARAMETERS)
        except RuntimeError as exc:
            raise VadBackendError(str(exc)) from exc
        return tuple(Interval(timestamp["start"], timestamp["end"]) for timestamp in timestamps)


def merge_regions(intervals: Sequence[Interval]) -> tuple[SpeechRegion, ...]:
    """Merge sorted, disjoint raw intervals across gaps <= 1 s, preserving activity."""
    regions: list[SpeechRegion] = []
    for interval in intervals:
        active_samples = interval.end - interval.start
        if regions and interval.start - regions[-1].interval.end <= MERGE_GAP_SAMPLES:
            previous = regions.pop()
            merged_interval = Interval(previous.interval.start, interval.end)
            merged_region = SpeechRegion(merged_interval, previous.active_samples + active_samples)
            regions.append(merged_region)
        else:
            regions.append(SpeechRegion(interval, active_samples))
    return tuple(regions)


def intersect_intervals(left: Sequence[Interval], right: Sequence[Interval]) -> tuple[Interval, ...]:
    """Intersect two sorted, disjoint raw interval lists in linear time."""
    intersections = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        first, second = left[left_index], right[right_index]
        start, end = max(first.start, second.start), min(first.end, second.end)
        if start < end:
            intersections.append(Interval(start, end))
        if first.end <= second.end:
            left_index += 1
        else:
            right_index += 1
    return tuple(intersections)


def utterance_regions(intervals: Sequence[Interval]) -> tuple[SpeechRegion, ...]:
    """Keep merged regions with >= 0.5 s raw activity, which also ensures span."""
    merged_regions = merge_regions(intervals)
    return tuple(region for region in merged_regions if region.active_samples >= UTTERANCE_MIN_ACTIVE_SAMPLES)


def overlap_regions(
    target: Sequence[Interval], residual: Sequence[Interval], sample_count: int
) -> tuple[SpeechRegion, ...]:
    """Intersect raw speech, merge, filter at 0.2 s activity, then pad to 0.5 s.

    Padding uses the right side for an odd extra sample and transfers unavailable
    boundary padding to the other side. If the recording is shorter than 0.5 s,
    retain the unpadded candidate: the scoring adapter must mark its short span
    ineligible so reporting preserves its candidate count.
    """
    intersections = intersect_intervals(target, residual)
    merged_regions = merge_regions(intersections)
    regions = []
    for region in merged_regions:
        if region.active_samples < OVERLAP_MIN_ACTIVE_SAMPLES:
            continue
        interval = region.interval
        padding = OVERLAP_MIN_SPAN_SAMPLES - (interval.end - interval.start)
        if padding > 0 and sample_count >= OVERLAP_MIN_SPAN_SAMPLES:
            latest_start = min(interval.start - padding // 2, sample_count - OVERLAP_MIN_SPAN_SAMPLES)
            start = max(0, latest_start)
            interval = Interval(start, start + OVERLAP_MIN_SPAN_SAMPLES)
        regions.append(SpeechRegion(interval, region.active_samples))
    return tuple(regions)


def activity_mask(intervals: Sequence[Interval], sample_count: int) -> NDArray[np.bool_]:
    """Mark strict-majority raw speech in 100 ms frames, using the last actual span.

    Inputs are sorted, disjoint raw intervals; merged gaps and overlap padding must not
    be supplied. Working memory scales with frames, rather than waveform samples.
    """
    frame_count = math.ceil(sample_count / ACTIVITY_FRAME_SAMPLES)
    active_samples = np.zeros(frame_count, dtype=np.int64)
    # Accumulate raw interval intersections with each frame, including fragments.
    for interval in intervals:
        first = interval.start // ACTIVITY_FRAME_SAMPLES
        last = (interval.end - 1) // ACTIVITY_FRAME_SAMPLES
        if first == last:
            active_samples[first] += interval.end - interval.start
        else:
            active_samples[first] += (first + 1) * ACTIVITY_FRAME_SAMPLES - interval.start
            active_samples[first + 1 : last] += ACTIVITY_FRAME_SAMPLES
            active_samples[last] += interval.end - last * ACTIVITY_FRAME_SAMPLES
    frame_starts = np.arange(frame_count) * ACTIVITY_FRAME_SAMPLES
    frame_spans = np.minimum(ACTIVITY_FRAME_SAMPLES, sample_count - frame_starts)
    return 2 * active_samples > frame_spans


def activity_counts(reference: NDArray[np.bool_], separated: NDArray[np.bool_]) -> ActivityCounts:
    """Compare equal-length full-channel masks, returning pooled confusion counts."""
    if reference.ndim != 1 or separated.shape != reference.shape:
        raise ValueError("Activity masks must be one-dimensional with equal lengths")
    true_positives = np.count_nonzero(reference & separated)
    false_positives = np.count_nonzero(~reference & separated)
    false_negatives = np.count_nonzero(reference & ~separated)
    return ActivityCounts(
        tp=int(true_positives), fp=int(false_positives), fn=int(false_negatives), frames=len(reference)
    )


@dataclass(frozen=True)
class ChannelSpeech:
    """Prepared raw activity, utterances, and frame mask for one full channel."""

    intervals: tuple[Interval, ...]
    utterances: tuple[SpeechRegion, ...]
    frames: NDArray[np.bool_]


@dataclass(frozen=True)
class ReferenceSpeech:
    """One recording's reusable reference timing; never share across mixtures."""

    sample_count: int
    target: ChannelSpeech
    residual: ChannelSpeech
    overlap: tuple[SpeechRegion, ...]


def _channel_speech(samples: NDArray[np.float32], detector: SpeechDetector) -> ChannelSpeech:
    """Prepare all timing populations from one independent full-channel VAD call."""
    intervals = detector(samples)
    frames = activity_mask(intervals, len(samples))
    frames.flags.writeable = False
    utterances = utterance_regions(intervals)
    return ChannelSpeech(intervals, utterances, frames)


def prepare_reference(reference: NDArray[np.float32], detector: SpeechDetector) -> ReferenceSpeech:
    """Prepare validated (2, n) reference audio once for its inference variants."""
    if reference.ndim != 2 or reference.shape[0] != 2 or reference.shape[1] == 0:
        raise ValueError("Reference VAD requires nonempty (2, n) audio")
    target, residual = (_channel_speech(samples, detector) for samples in reference)
    sample_count = reference.shape[1]
    overlap = overlap_regions(target.intervals, residual.intervals, sample_count)
    return ReferenceSpeech(sample_count, target, residual, overlap)


def prepare_separation(
    separated: NDArray[np.float32], reference: ReferenceSpeech, detector: SpeechDetector
) -> dict[Channel, ChannelResult]:
    """Prepare validated (1, n)/(2, n) output against reusable reference timing.

    Return initial T/R records with VAD and activity populated. Missing R output
    has unavailable activity while retaining reference timing. Other metrics stay
    disabled for the pipeline to fill according to its selected configuration.
    """
    if separated.ndim != 2 or separated.shape[0] not in (1, 2) or separated.shape[1] != reference.sample_count:
        raise ValueError("Separated VAD requires (1, n) or (2, n) audio matching the reference length")
    channels: dict[Channel, ChannelResult] = {}
    # Reference calls are deliberately outside this loop and reused for every variant.
    for index, (channel, original) in enumerate((("T", reference.target), ("R", reference.residual))):
        available = index < separated.shape[0]
        output = _channel_speech(separated[index], detector) if available else None
        if output is not None:
            counts = activity_counts(original.frames, output.frames)
            activity = ActivityResult("scored", counts)
        else:
            activity = ActivityResult("unavailable", reason="missing_output_channel")
        channels[channel] = ChannelResult(
            status="available" if available else "unavailable",
            reference_active=bool(original.intervals),
            reference_vad=original.intervals,
            reference_utterances=original.utterances,
            separated_vad=output.intervals if output is not None else (),
            separated_utterances=output.utterances if output is not None else (),
            activity=activity,
        )
    return channels
