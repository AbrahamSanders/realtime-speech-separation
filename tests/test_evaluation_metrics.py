"""Core metric adapter contracts with injected functions; native SI-SDR is opt-in."""

import os
import sys
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from realtime_speech_separation.evaluation import metrics
from realtime_speech_separation.evaluation.pipeline import _interval_result
from realtime_speech_separation.evaluation.types import Interval


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ordinary adapter tests free of optional runtime imports."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test", from_numpy=np.asarray))
    monkeypatch.setattr(metrics, "version", lambda name: "test")


def test_si_sdr_argument_order_dtype_and_zero_mean(fake_runtime: None) -> None:
    """The functional API takes prediction first while the adapter takes reference first."""
    reference = np.arange(8, dtype=np.float32)
    estimate = reference[::-1]
    call = Mock(return_value=np.float32(6))
    adapter = metrics.SiSdr(call)
    score = adapter(reference, estimate)
    assert score == metrics.MetricScore("scored", 6.0)
    prediction, target = call.call_args.args
    np.testing.assert_array_equal(prediction, estimate)
    np.testing.assert_array_equal(target, reference)
    assert prediction.dtype == target.dtype == np.float32
    assert prediction.flags.c_contiguous
    assert call.call_args.kwargs == {"zero_mean": True}
    assert adapter.parameters["epsilon"] == 2**-23


@pytest.mark.parametrize("constant", [0, 0.1, -2])
def test_centered_silence_and_quiet_nonzero(fake_runtime: None, constant: float) -> None:
    """Constant offsets are zero energy; quiet changing estimates have no cutoff."""
    reference = np.linspace(-1, 1, 8000, dtype=np.float32)
    silence = np.full(8000, constant, dtype=np.float32)
    call = Mock(return_value=2)
    adapter = metrics.SiSdr(call)
    assert adapter(reference, silence).status == "undefined"
    assert adapter(silence, reference).status == "ineligible"
    call.assert_not_called()
    quiet = reference * np.float32(1e-30)
    assert adapter(reference, quiet).value == 2
    call.assert_called_once()


def test_si_sdr_failures_and_unexpected_errors(fake_runtime: None) -> None:
    """Runtime/nonfinite scores fail locally; programming errors remain visible."""
    samples = np.arange(8, dtype=np.float32)
    for value in (np.nan, np.inf, -np.inf):
        assert metrics.SiSdr(Mock(return_value=value))(samples, samples).reason == "nonfinite_score"
    failed = metrics.SiSdr(Mock(side_effect=RuntimeError("native failure")))(samples, samples)
    assert failed.status == "failed"
    assert "native failure" in failed.reason
    with pytest.raises(TypeError, match="bug"):
        metrics.SiSdr(Mock(side_effect=TypeError("bug")))(samples, samples)


def test_stoi_reference_first_valid_zero_and_warning(fake_runtime: None) -> None:
    """Warning identity, rather than score magnitude, identifies insufficient speech."""
    reference = np.arange(8000, dtype=np.float32)
    estimate = -reference
    call = Mock(return_value=0)
    adapter = metrics.Stoi(call)
    assert adapter(reference, estimate) == metrics.MetricScore("scored", 0.0)
    args = call.call_args.args
    assert args[0] is reference and args[1] is estimate and args[2] == 16000
    assert call.call_args.kwargs == {"extended": False}

    def insufficient(*args, **kwargs):
        """Mimic pystoi's documented warning and numerical sentinel."""
        warnings.warn(
            "Not enough STFT frames to compute intermediate intelligibility measure", RuntimeWarning, stacklevel=2
        )
        return 1e-5

    unavailable = metrics.Stoi(insufficient)(reference, estimate)
    assert unavailable.status == "ineligible"
    assert unavailable.reason == "insufficient_stoi_frames"
    assert metrics.Stoi(Mock(return_value=1e-5))(reference, estimate).status == "scored"
    assert metrics.Stoi(Mock(return_value=np.nan))(reference, estimate).status == "failed"


def test_other_stoi_warnings_and_programming_errors_escape(fake_runtime: None) -> None:
    """Only the documented insufficient-frame warning has special treatment."""
    samples = np.arange(8, dtype=np.float32)

    def unrelated(*args, **kwargs):
        """Emit a warning that must remain visible to callers."""
        warnings.warn("some other warning", RuntimeWarning, stacklevel=2)
        return 0.2

    with pytest.warns(RuntimeWarning, match="some other warning"):
        assert metrics.Stoi(unrelated)(samples, samples).value == 0.2
    with pytest.raises(ValueError, match="bug"):
        metrics.Stoi(Mock(side_effect=ValueError("bug")))(samples, samples)


@pytest.mark.parametrize(
    ("baseline", "status", "input_score"),
    [
        (metrics.MetricScore("scored", -2), "scored", -2),
        (metrics.MetricScore("undefined", reason="zero_energy_estimate"), "undefined", None),
        (metrics.MetricScore("ineligible", reason="insufficient_stoi_frames"), "undefined", None),
        (metrics.MetricScore("failed", reason="nonfinite_score"), "failed", None),
    ],
)
def test_paired_scores_preserve_output_when_baseline_unavailable(baseline, status, input_score) -> None:
    """An unavailable mixture baseline does not erase a valid separated score."""
    result = _interval_result(Interval(0, 8000), "reference_utterance", metrics.MetricScore("scored", 3), baseline)
    assert result.status == status
    assert result.input_score == input_score
    assert result.output_score == 3
    assert result.value == (5 if status == "scored" else None)


@pytest.mark.metric_backend
@pytest.mark.skipif(os.environ.get("RTSS_TEST_METRIC_BACKENDS") != "1", reason="opt-in installed SI-SDR backend")
def test_real_si_sdr_mixture_as_estimate_has_zero_improvement() -> None:
    """Exercise the actual float32 functional implementation on CPU only."""
    rng = np.random.default_rng(42)
    reference = rng.normal(size=16000).astype(np.float32)
    noise = rng.normal(scale=0.4, size=16000).astype(np.float32)
    mixture = reference + noise
    adapter = metrics.SiSdr()
    baseline = adapter(reference, mixture)
    output = adapter(reference, mixture.copy())
    result = _interval_result(Interval(0, 16000), "reference_utterance", output, baseline)
    assert result.status == "scored"
    assert result.value == 0
