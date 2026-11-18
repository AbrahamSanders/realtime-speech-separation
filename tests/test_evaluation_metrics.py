"""Metric adapter contracts with fake runtimes and opt-in native smoke tests."""

import os
import sys
import warnings
from hashlib import sha256
from pathlib import Path
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


@pytest.fixture
def fake_pesq(monkeypatch: pytest.MonkeyPatch, fake_runtime: None) -> SimpleNamespace:
    """Expose the wrapper's callable and exception hierarchy without loading native code."""

    class PesqError(RuntimeError):
        """Match the wrapper's base exception and error-mode constant."""

        RAISE_EXCEPTION = 0

    class BufferTooShortError(PesqError):
        """Represent an unsupported buffer duration."""

    class NoUtterancesError(PesqError):
        """Represent the lack of a detectable utterance."""

    backend = SimpleNamespace(
        pesq=Mock(return_value=np.float32(3.25)),
        PesqError=PesqError,
        BufferTooShortError=BufferTooShortError,
        NoUtterancesError=NoUtterancesError,
    )
    monkeypatch.setitem(sys.modules, "pesq", backend)
    return backend


def test_pesq_wideband_arguments_and_provenance(fake_pesq: SimpleNamespace) -> None:
    """Pass unmodified mono waveform slices to the wideband exception-mode API."""
    reference = np.linspace(-1, 1, 8000, dtype=np.float32)
    estimate = reference[::-1]
    adapter = metrics.Pesq()
    assert adapter(reference, estimate) == metrics.MetricScore("scored", 3.25)
    call = fake_pesq.pesq
    sample_rate, ref_arg, est_arg = call.call_args.args
    assert sample_rate == 16000 and ref_arg is reference and est_arg is estimate
    assert call.call_args.kwargs == {"mode": "wb", "on_error": fake_pesq.PesqError.RAISE_EXCEPTION}
    assert adapter.parameters == {"sample_rate": 16000, "mode": "wb", "on_error": "raise_exception"}
    assert adapter.versions == {"pesq": "test"}


@pytest.mark.parametrize(
    ("error_type", "status", "reason"),
    [
        ("BufferTooShortError", "ineligible", "pesq_buffer_too_short"),
        ("NoUtterancesError", "ineligible", "pesq_no_utterances"),
        ("PesqError", "failed", "pesq_backend_error: native error"),
    ],
)
def test_pesq_typed_native_errors(fake_pesq: SimpleNamespace, error_type: str, status: str, reason: str) -> None:
    """Keep documented eligibility exclusions distinct from scoring failures."""
    samples = np.arange(8000, dtype=np.float32)
    fake_pesq.pesq.side_effect = getattr(fake_pesq, error_type)("native error")
    assert metrics.Pesq()(samples, samples) == metrics.MetricScore(status, reason=reason)


def test_pesq_nonfinite_scores_and_unexpected_errors(fake_pesq: SimpleNamespace) -> None:
    """Handle the wrapper's native NaN conversion failure while exposing other bugs."""
    samples = np.arange(8000, dtype=np.float32)
    adapter = metrics.Pesq()
    failed = metrics.MetricScore("failed", reason="nonfinite_score")
    for value in (np.nan, np.inf, -np.inf):
        fake_pesq.pesq.return_value = value
        assert adapter(samples, samples) == failed
    fake_pesq.pesq.side_effect = ValueError("cannot convert float NaN to integer")
    assert adapter(samples, samples) == failed
    fake_pesq.pesq.side_effect = ValueError("unexpected error")
    with pytest.raises(ValueError, match="unexpected error"):
        adapter(samples, samples)


@pytest.fixture
def fake_visqol(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_runtime: None) -> SimpleNamespace:
    """Stand in for the official binding, generated protobufs, and local assets."""

    class StatusNotOk(Exception):
        """Match the native binding's Abseil status exception."""

    model_path = tmp_path / "model" / "speech.tflite"
    model_path.parent.mkdir()
    model_path.write_bytes(b"speech lattice model")
    extension = tmp_path / "visqol_lib_py.so"
    extension.write_bytes(b"native library")
    api = SimpleNamespace(Create=Mock(), Measure=Mock(return_value=SimpleNamespace(moslqo=3.5)))
    config = SimpleNamespace(audio=SimpleNamespace(), options=SimpleNamespace())
    native = SimpleNamespace(
        __file__=str(extension), DefaultSpeechModelFile=lambda: "/model/speech.tflite", VisqolApi=Mock(return_value=api)
    )
    protobuf = SimpleNamespace(VisqolConfig=Mock(return_value=config))
    monkeypatch.setitem(sys.modules, "visqol", SimpleNamespace(visqol_lib_py=native))
    monkeypatch.setitem(sys.modules, "visqol.pb2", SimpleNamespace(visqol_config_pb2=protobuf))
    monkeypatch.setitem(sys.modules, "visqol.pb2.similarity_result_pb2", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pybind11_abseil.status", SimpleNamespace(StatusNotOk=StatusNotOk))
    return SimpleNamespace(api=api, config=config, model_path=model_path, extension=extension, status_error=StatusNotOk)


def test_visqol_speech_configuration(fake_visqol: SimpleNamespace) -> None:
    """Use the packaged lattice model and record the entire input policy."""
    adapter = metrics.Visqol()
    config = fake_visqol.config
    fake_visqol.api.Create.assert_called_once_with(config)
    assert config.audio.sample_rate == 16000
    assert config.options.use_speech_scoring is config.options.use_lattice_model is True
    assert config.options.search_window_radius == 60
    assert config.options.svr_model_path == str(fake_visqol.model_path)
    assert adapter.model_digest == sha256(fake_visqol.model_path.read_bytes()).hexdigest()
    assert adapter.parameters["extension_sha256"] == sha256(fake_visqol.extension.read_bytes()).hexdigest()
    assert adapter.versions == {"visqol": "test", "protobuf": "test"}
    assert adapter.parameters["min_span_samples"] == 32000
    assert adapter.parameters["max_span_samples"] == 144000
    assert adapter.parameters["edge_padding_samples"] == 8000


@pytest.mark.parametrize(("size", "scored_samples"), [(32000, 32000), (144000, 144000), (160000, 144000)])
def test_visqol_trims_and_pads_aligned_inputs(fake_visqol: SimpleNamespace, size: int, scored_samples: int) -> None:
    """Score the same prefix in both signals and add exactly half a second at each edge."""
    adapter = metrics.Visqol()
    reference = np.linspace(-1, 1, size, dtype=np.float32)
    estimate = reference[::-1]
    short = metrics.MetricScore("ineligible", reason="visqol_buffer_too_short")
    assert adapter(reference[:31999], estimate[:31999]) == short
    fake_visqol.api.Measure.assert_not_called()
    assert adapter(reference, estimate) == metrics.MetricScore("scored", 3.5)
    ref_arg, est_arg = fake_visqol.api.Measure.call_args.args
    for padded, original in ((ref_arg, reference), (est_arg, estimate)):
        assert padded.size == scored_samples + 16000
        np.testing.assert_array_equal(padded[8000:-8000], original[:scored_samples])
        assert not np.any(padded[:8000]) and not np.any(padded[-8000:])
        assert padded.dtype == np.float64 and padded.flags.c_contiguous


def test_visqol_native_failures_and_nonfinite_scores(fake_visqol: SimpleNamespace) -> None:
    """Fail preflight for model loading errors and isolate failures during scoring."""
    fake_visqol.api.Create.side_effect = fake_visqol.status_error("invalid model")
    with pytest.raises(RuntimeError, match="ViSQOL could not initialize.*invalid model"):
        metrics.Visqol()
    fake_visqol.api.Create.side_effect = TypeError("incompatible function arguments")
    with pytest.raises(RuntimeError, match="compatible protobuf versions"):
        metrics.Visqol()
    fake_visqol.api.Create.side_effect = None
    adapter = metrics.Visqol()
    samples = np.ones(32000, dtype=np.float32)
    fake_visqol.api.Measure.return_value = SimpleNamespace(moslqo=np.nan)
    assert adapter(samples, samples) == metrics.MetricScore("failed", reason="nonfinite_score")
    for error in (
        fake_visqol.status_error("scoring failed"),
        RuntimeError("scoring failed"),
        IndexError("scoring failed"),
    ):
        fake_visqol.api.Measure.side_effect = error
        assert adapter(samples, samples) == metrics.MetricScore("failed", reason="visqol_backend_error: scoring failed")
    fake_visqol.api.Measure.side_effect = TypeError("unexpected bug")
    with pytest.raises(TypeError, match="unexpected bug"):
        adapter(samples, samples)


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


@pytest.mark.metric_backend
@pytest.mark.skipif(os.environ.get("RTSS_TEST_METRIC_BACKENDS") != "1", reason="opt-in installed metric backends")
def test_real_pesq_scores_signal_and_reports_native_exceptions() -> None:
    """Exercise the native wrapper using a deterministic modulated signal."""
    rng = np.random.default_rng(42)
    time = np.arange(32000) / 16000
    envelope = np.maximum(np.sin(2 * np.pi * 3 * time), 0)
    carrier = np.sin(2 * np.pi * 190 * time) + 0.3 * rng.standard_normal(time.size)
    signal = (envelope * carrier).astype(np.float32)
    silence = np.zeros_like(signal)
    adapter = metrics.Pesq()
    score = adapter(signal, signal.copy())
    assert score.status == "scored"
    assert score.value == pytest.approx(4.64, abs=0.02)
    assert adapter(signal[:3200], signal[:3200]) == metrics.MetricScore("ineligible", reason="pesq_buffer_too_short")
    assert adapter(silence, signal) == metrics.MetricScore("ineligible", reason="pesq_no_utterances")
    assert adapter(signal, silence) == metrics.MetricScore("failed", reason="nonfinite_score")


@pytest.mark.metric_backend
@pytest.mark.skipif(os.environ.get("RTSS_TEST_VISQOL") != "1", reason="opt-in installed ViSQOL backend")
def test_real_visqol_speech_minimum_and_mos_lqo() -> None:
    """Verify the native minimum and the padded protocol inputs on the real API."""
    rng = np.random.default_rng(42)
    time = np.arange(160000) / 16000
    envelope = 0.4 * (0.65 + 0.35 * np.sin(2 * np.pi * 3 * time))
    carrier = np.sin(2 * np.pi * 190 * time) + 0.3 * rng.standard_normal(time.size)
    signal = (envelope * carrier).astype(np.float32)
    adapter = metrics.Visqol()
    for size in (8000, 10239):
        samples = signal[:size]
        native_samples = samples.astype(np.float64)
        with pytest.raises(IndexError, match=r"Mat::rows\(\): indices out of bounds"):
            adapter._api.Measure(native_samples, native_samples)
    native_boundary = signal[:10240].astype(np.float64)
    assert 4.0 <= adapter._api.Measure(native_boundary, native_boundary).moslqo <= 5.0
    too_short = signal[:31999]
    assert adapter(too_short, too_short) == metrics.MetricScore("ineligible", reason="visqol_buffer_too_short")
    for size in (32000, 144000, 160000):
        samples = signal[:size]
        score = adapter(samples, samples)
        assert score.status == "scored"
        assert 4.0 <= score.value <= 5.0
