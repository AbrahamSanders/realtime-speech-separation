"""Offline VAD timing boundaries and resource reuse with an injected Silero runtime."""

import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from realtime_speech_separation.evaluation import vad
from realtime_speech_separation.evaluation.types import ActivityCounts, Interval, SpeechRegion


@pytest.mark.parametrize("gap", [0, 15999, 16000, 16001])
def test_merge_gap_boundary_preserves_activity(gap: int) -> None:
    """Exactly one second merges, without counting any intervening silence."""
    intervals = (Interval(0, 4000), Interval(4000 + gap, 8000 + gap))
    expected = (
        (SpeechRegion(Interval(0, 8000 + gap), 8000),)
        if gap <= 16000
        else (SpeechRegion(intervals[0], 4000), SpeechRegion(intervals[1], 4000))
    )
    assert vad.merge_regions(intervals) == expected


@pytest.mark.parametrize("active_samples", [7999, 8000, 8001])
def test_utterance_inclusive_active_minimum(active_samples: int) -> None:
    """Use active duration, including at the exact 0.5 s boundary."""
    intervals = (Interval(0, 4000), Interval(20000, 20000 + active_samples - 4000))
    regions = vad.utterance_regions(intervals)
    assert regions == (
        (SpeechRegion(Interval(0, 16000 + active_samples), active_samples),) if active_samples >= 8000 else ()
    )


def test_long_span_with_too_little_speech_is_not_an_utterance() -> None:
    """Chained short fragments cannot qualify just by spanning silent gaps."""
    intervals = tuple(Interval(start, start + 1000) for start in (0, 16000, 32000, 48000))
    assert vad.merge_regions(intervals) == (SpeechRegion(Interval(0, 49000), 4000),)
    assert vad.utterance_regions(intervals) == ()


def test_intersection_half_open_bounds_and_fragments() -> None:
    """Intersections use raw speech and handle matching ends and fragmented inputs."""
    target = (Interval(0, 100), Interval(150, 300), Interval(400, 500))
    residual = (Interval(100, 150), Interval(200, 250), Interval(260, 300), Interval(350, 450))
    assert vad.intersect_intervals(target, residual) == (Interval(200, 250), Interval(260, 300), Interval(400, 450))
    assert vad.intersect_intervals(residual, target) == vad.intersect_intervals(target, residual)
    assert vad.intersect_intervals((), residual) == ()


def test_overlap_does_not_intersect_merged_silence() -> None:
    """Utterance spans can overlap even when the underlying speakers never do."""
    target = (Interval(0, 4000), Interval(12000, 16000))
    residual = (Interval(4000, 12000),)
    assert vad.utterance_regions(target)
    assert vad.utterance_regions(residual)
    assert vad.overlap_regions(target, residual, 16000) == ()


@pytest.mark.parametrize("active_samples", [3199, 3200, 3201])
def test_overlap_active_threshold_and_padding(active_samples: int) -> None:
    """Fragments merge at the exact gap limit; padding never adds activity."""
    target = (Interval(0, 1600), Interval(17600, 17600 + active_samples - 1600))
    residual = (Interval(0, 24000),)
    assert vad.overlap_regions(target, residual, 24000) == (
        (SpeechRegion(Interval(0, 16000 + active_samples), active_samples),) if active_samples >= 3200 else ()
    )


@pytest.mark.parametrize(
    ("interval", "sample_count", "expected"),
    [
        (Interval(0, 3200), 16000, Interval(0, 8000)),
        (Interval(12800, 16000), 16000, Interval(8000, 16000)),
        (Interval(4000, 7200), 16000, Interval(1600, 9600)),
        (Interval(4000, 7201), 16000, Interval(1601, 9601)),
        (Interval(1000, 4200), 8000, Interval(0, 8000)),
        (Interval(0, 8000), 8000, Interval(0, 8000)),
        (Interval(0, 10000), 16000, Interval(0, 10000)),
        (Interval(0, 3200), 7999, Interval(0, 3200)),
    ],
)
def test_overlap_padding_boundaries(interval: Interval, sample_count: int, expected: Interval) -> None:
    """Redistribute boundary padding; retain short-recording ineligible candidates."""
    assert vad.overlap_regions((interval,), (interval,), sample_count) == (
        SpeechRegion(expected, interval.end - interval.start),
    )


def test_no_speech_and_no_overlap() -> None:
    """Empty raw speech produces no utterances or overlap candidates."""
    assert vad.merge_regions(()) == ()
    assert vad.utterance_regions(()) == ()
    assert vad.overlap_regions((), (), 16000) == ()
    assert vad.overlap_regions((Interval(0, 8000),), (Interval(8000, 16000),), 16000) == ()


@pytest.mark.parametrize(
    ("intervals", "sample_count", "expected"),
    [
        ((), 3200, [False, False]),
        ((Interval(0, 800),), 1600, [False]),
        ((Interval(0, 801),), 1600, [True]),
        ((Interval(0, 400), Interval(800, 1201)), 1600, [True]),
        ((Interval(800, 2400),), 3200, [False, False]),
        ((Interval(799, 2401),), 3200, [True, True]),
        ((Interval(100, 5000),), 5000, [True, True, True, True]),
        ((Interval(3200, 3300),), 3400, [False, False, False]),
        ((Interval(3200, 3301),), 3400, [False, False, True]),
        ((Interval(3200, 3201),), 3201, [False, False, True]),
    ],
)
def test_activity_strict_majority_and_partial_frame(
    intervals: tuple[Interval, ...], sample_count: int, expected: list[bool]
) -> None:
    """Use raw fragment totals, strict ties, and the actual final frame length."""
    mask = vad.activity_mask(intervals, sample_count)
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, expected)


def test_activity_counts_include_all_frames() -> None:
    """TP/FP/FN omit true negatives while retaining the complete frame denominator."""
    reference = np.array([True, True, False, False, True])
    separated = np.array([True, False, True, False, True])
    assert vad.activity_counts(reference, separated) == ActivityCounts(2, 1, 1, 5)
    silence = np.zeros(5, dtype=bool)
    assert vad.activity_counts(silence, silence) == ActivityCounts(0, 0, 0, 5)
    with pytest.raises(ValueError, match="equal lengths"):
        vad.activity_counts(reference, separated[:2])


@pytest.fixture
def fake_silero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """Install a small fake runtime; no optional packages, model, or GPU are used."""
    state = SimpleNamespace(loads=0, resets=0, calls=[], error=None)
    state.outputs = {
        1: [{"start": 0, "end": 8000}],
        2: [],
        3: [{"start": 0, "end": 4000}],
        4: [{"start": 8000, "end": 12000}],
        5: [],
    }

    class Model:
        """Recurrent state must be explicitly reset before every channel call."""

        dirty = True

        def cpu(self) -> "Model":
            """Record explicit CPU selection."""
            state.cpu = True
            return self

        def eval(self) -> "Model":
            """Record evaluation mode selection."""
            state.eval = True
            return self

        def reset_states(self) -> None:
            """Clear the recurrent state left by the preceding channel."""
            state.resets += 1
            self.dirty = False

    def load_silero_vad(*, onnx: bool) -> Model:
        """Count loads and require the packaged JIT model selection."""
        assert onnx is False
        state.loads += 1
        return Model()

    def get_speech_timestamps(samples: np.ndarray, model: Model, **kwargs: object) -> list[dict[str, int]]:
        """Check independent state, protocol settings, and untouched full channels."""
        model.reset_states()
        assert not model.dirty
        assert samples.shape == (16000,)
        assert samples.dtype == np.float32
        assert samples.flags.c_contiguous
        assert kwargs == {
            "sampling_rate": 16000,
            "threshold": 0.5,
            "neg_threshold": 0.35,
            "min_speech_duration_ms": 0,
            "min_silence_duration_ms": 100,
            "speech_pad_ms": 0,
            "return_seconds": False,
        }
        model.dirty = True
        marker = int(samples[0])
        state.calls.append(marker)
        if state.error:
            raise state.error
        return state.outputs[marker]

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            from_numpy=lambda samples: samples,
            __version__="test-torch",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "silero_vad",
        SimpleNamespace(load_silero_vad=load_silero_vad, get_speech_timestamps=get_speech_timestamps),
    )
    (tmp_path / "silero_vad.jit").write_bytes(b"packaged-vad-model")
    monkeypatch.setattr(vad, "files", lambda package: tmp_path)
    monkeypatch.setattr(vad, "version", lambda package: "6.2.1")
    return state


def test_silero_channel_calls_state_reset_and_reference_reuse(fake_silero: SimpleNamespace) -> None:
    """Load once; reuse T/R reference timing across stereo and target-only variants."""
    detector = vad.SileroVad()
    reference = np.stack([np.full(16000, marker, dtype=np.float32) for marker in (1, 2)])
    separated = np.stack([np.full(16000, marker, dtype=np.float32) for marker in (3, 4)])
    original = vad.prepare_reference(reference, detector)
    channels = vad.prepare_separation(separated, original, detector)
    target_only = vad.prepare_separation(np.full((1, 16000), 5, dtype=np.float32), original, detector)
    vad.prepare_reference(reference, detector)

    assert fake_silero.loads == 1
    assert fake_silero.calls == [1, 2, 3, 4, 5, 1, 2]
    assert fake_silero.resets == 7
    assert fake_silero.cpu and fake_silero.eval
    assert detector.backend_version == "6.2.1"
    assert detector.torch_version == "test-torch"
    assert detector.model_digest == sha256(b"packaged-vad-model").hexdigest()
    assert detector.parameters["neg_threshold"] == 0.35
    assert detector.parameters["device"] == "cpu"
    assert "threads" not in detector.parameters
    assert original.target.frames.flags.writeable is False
    assert channels["T"].reference_vad is target_only["T"].reference_vad
    assert channels["T"].reference_utterances == (SpeechRegion(Interval(0, 8000), 8000),)
    assert channels["T"].separated_utterances == ()
    assert channels["T"].activity.counts == ActivityCounts(2, 0, 3, 10)
    assert channels["R"].reference_active is False
    assert channels["R"].activity.counts == ActivityCounts(0, 2, 0, 10)
    assert target_only["T"].activity.counts == ActivityCounts(0, 0, 5, 10)
    assert target_only["R"].status == "unavailable"
    assert target_only["R"].activity.status == "unavailable"
    assert target_only["R"].activity.reason == "missing_output_channel"
    assert original.overlap == ()


@pytest.mark.parametrize("exception_type", [RuntimeError, TypeError])
def test_vad_runtime_failure_and_unexpected_errors(
    fake_silero: SimpleNamespace, exception_type: type[Exception]
) -> None:
    """Wrap runtime execution failures, while letting programming errors propagate."""
    detector = vad.SileroVad()
    fake_silero.error = exception_type("backend failure")
    expected = vad.VadBackendError if exception_type is RuntimeError else TypeError
    with pytest.raises(expected, match="backend failure"):
        detector(np.full(16000, 1, dtype=np.float32))
    fake_silero.error = None
    assert detector(np.full(16000, 1, dtype=np.float32)) == (Interval(0, 8000),)
    assert fake_silero.resets == 2


def test_preparation_keeps_independent_channel_timing() -> None:
    """Both reference channels run even when a target-only output omits residual."""
    intervals = {
        0: (),
        1: (Interval(0, 4000), Interval(8000, 12000)),
        2: (Interval(2000, 6000), Interval(10000, 14000)),
    }

    def detector(samples: np.ndarray) -> tuple[Interval, ...]:
        """Inject already validated full-channel speech populations."""
        return intervals[int(samples[0])]

    reference = np.stack([np.full(16000, marker, dtype=np.float32) for marker in (1, 2)])
    original = vad.prepare_reference(reference, detector)
    channels = vad.prepare_separation(np.zeros((1, 16000), dtype=np.float32), original, detector)
    assert original.overlap == (SpeechRegion(Interval(2000, 12000), 4000),)
    assert channels["T"].activity.counts == ActivityCounts(0, 0, 4, 10)
    assert channels["R"].reference_vad == intervals[2]
    assert channels["R"].reference_active is True
    assert channels["R"].activity.status == "unavailable"


def test_preparation_requires_loaded_audio_shapes() -> None:
    """Reject caller shape or length mistakes before invoking a detector."""

    def detector(samples: np.ndarray) -> tuple[Interval, ...]:
        """Return no speech without using an optional runtime."""
        return ()

    with pytest.raises(ValueError, match="Reference VAD"):
        vad.prepare_reference(np.zeros((1, 16000), dtype=np.float32), detector)
    original = vad.prepare_reference(np.zeros((2, 16000), dtype=np.float32), detector)
    with pytest.raises(ValueError, match="matching the reference length"):
        vad.prepare_separation(np.zeros((1, 15999), dtype=np.float32), original, detector)
