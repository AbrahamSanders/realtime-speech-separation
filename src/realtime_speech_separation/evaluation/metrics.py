"""Lazy CPU metric adapters for the evaluation pipeline.

Adapters expose scoring settings in `parameters`. The optional
`min_active_samples` sets the pipeline's minimum detected speech in a region.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .audio import EVALUATION_SAMPLE_RATE
from .reporting import content_digest
from .types import IntervalResultStatus
from .vad import OVERLAP_MIN_ACTIVE_SAMPLES

MODEL_QUALITY_MIN_ACTIVE_SAMPLES = 3 * EVALUATION_SAMPLE_RATE


@dataclass(frozen=True)
class MetricScore:
    """An adapter outcome without interval bounds, also used for mixture baselines."""

    status: IntervalResultStatus
    value: float | None = None
    reason: str | None = None


def finite_score(value: float) -> MetricScore:
    """Convert backend scalars to ordinary finite floats for strict result JSON."""
    value = float(value)
    if not np.isfinite(value):
        return MetricScore("failed", reason="nonfinite_score")
    return MetricScore("scored", value)


def centered_zero_energy(samples: NDArray[np.float32]) -> bool:
    """Test exact centered silence without squaring away quiet nonzero samples.

    Float64 centering avoids rounding a constant float32 channel's mean away
    from its sample value. No positive energy threshold is applied.
    """
    centered = samples.astype(np.float64) - samples.mean(dtype=np.float64)
    has_energy = np.any(centered)
    return not bool(has_energy)


class SiSdr:
    """TorchMetrics functional SI-SDR, receiving (reference, estimate) at 16 kHz.

    Inputs are equal-length mono float32 arrays. Centered zero-energy references
    are ineligible and centered zero-energy estimates are undefined. Runtime
    scoring errors become failed outcomes; unexpected programming errors escape.
    The optional call is injectable for offline tests.
    """

    def __init__(self, call: Callable | None = None) -> None:
        """Resolve the optional functional runtime once, using CPU tensors."""
        import torch

        if call is None:
            from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

            call = scale_invariant_signal_distortion_ratio
        self._torch = torch
        self._call = call
        self.versions = {"torch": str(torch.__version__), "torchmetrics": version("torchmetrics")}
        dtype = "float32"
        self.parameters = {
            "zero_mean": True,
            "dtype": dtype,
            "device": "cpu",
            "min_active_samples": OVERLAP_MIN_ACTIVE_SAMPLES,
            "epsilon": float(np.finfo(dtype).eps),
            "epsilon_placement": "projection_numerator_denominator_and_both_energy_sums",
            "centered_zero_energy": "exact_float64_centering_without_positive_cutoff",
        }

    def __call__(self, reference: NDArray[np.float32], estimate: NDArray[np.float32]) -> MetricScore:
        """Preserve undefined silence rather than accepting epsilon placeholders."""
        if centered_zero_energy(reference):
            return MetricScore("ineligible", reason="zero_energy_reference")
        if centered_zero_energy(estimate):
            return MetricScore("undefined", reason="zero_energy_estimate")
        contiguous_reference = np.ascontiguousarray(reference, dtype=self.parameters["dtype"])
        contiguous_estimate = np.ascontiguousarray(estimate, dtype=self.parameters["dtype"])
        device = self.parameters["device"]
        target = self._torch.as_tensor(contiguous_reference, device=device)
        prediction = self._torch.as_tensor(contiguous_estimate, device=device)
        try:
            score = self._call(prediction, target, zero_mean=self.parameters["zero_mean"])
        except RuntimeError as exc:
            return MetricScore("failed", reason=f"si_sdr_backend_error: {exc}")
        return finite_score(score)


class Stoi:
    """Classical pystoi with reference first and no additional signal processing.

    The documented insufficient-STFT-frame warning is ineligible, independently
    of the returned numerical sentinel. Other warnings retain normal handling;
    a genuine zero score is valid. Inputs are equal-length mono float32 arrays.
    """

    def __init__(self, call: Callable | None = None) -> None:
        """Resolve the optional function once; allow injected calls in tests."""
        if call is None:
            from pystoi import stoi

            call = stoi
        self._call = call
        self.versions = {name: version(name) for name in ("pystoi", "scipy", "numpy")}
        self.parameters = {"sample_rate": EVALUATION_SAMPLE_RATE, "extended": False}

    def __call__(self, reference: NDArray[np.float32], estimate: NDArray[np.float32]) -> MetricScore:
        """Exclude the library's insufficient-speech sentinel using its warning."""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", message="Not enough STFT frames.*", category=RuntimeWarning)
                score = self._call(
                    reference, estimate, self.parameters["sample_rate"], extended=self.parameters["extended"]
                )
        except RuntimeWarning as exc:
            if str(exc).startswith("Not enough STFT frames"):
                return MetricScore("ineligible", reason="insufficient_stoi_frames")
            raise
        return finite_score(score)


class Pesq:
    """Wideband PESQ MOS-LQO for aligned mono float32 reference and estimate slices.

    Unusable reference speech and short buffers are ineligible. Native scoring
    errors and nonfinite scores are failed outcomes.
    """

    def __init__(self) -> None:
        """Resolve the PESQ wrapper and its typed errors."""
        import pesq

        self._backend = pesq
        self.versions = {"pesq": version("pesq")}
        self.parameters = {"sample_rate": EVALUATION_SAMPLE_RATE, "mode": "wb", "on_error": "RAISE_EXCEPTION"}

    def __call__(self, reference: NDArray[np.float32], estimate: NDArray[np.float32]) -> MetricScore:
        """Score a reference utterance using the wrapper's typed error mode."""
        backend = self._backend
        error_mode = getattr(backend.PesqError, self.parameters["on_error"])
        try:
            score = backend.pesq(
                self.parameters["sample_rate"], reference, estimate, mode=self.parameters["mode"], on_error=error_mode
            )
        except backend.BufferTooShortError:
            return MetricScore("ineligible", reason="pesq_buffer_too_short")
        except backend.NoUtterancesError:
            return MetricScore("ineligible", reason="pesq_no_utterances")
        except backend.PesqError as exc:
            return MetricScore("failed", reason=f"pesq_backend_error: {exc}")
        except ValueError as exc:
            # pesq 0.0.4 tries to decode a native NaN as an integer error code.
            if str(exc) == "cannot convert float NaN to integer":
                return MetricScore("failed", reason="nonfinite_score")
            raise
        return finite_score(score)


class Visqol:
    """Official ViSQOL speech-mode MOS-LQO for aligned mono float32 slices.

    The pipeline requires three seconds of detected reference speech. Eligible
    inputs use up to nine seconds of the utterance and half a second of silence
    at each edge. Native scoring errors and nonfinite scores are failed outcomes.
    """

    def __init__(self) -> None:
        """Load the installed CPU binding and its bundled speech model once."""
        try:
            from visqol import visqol_lib_py
            from visqol.pb2 import visqol_config_pb2

            # Register the result protobuf and retrieve the status type installed by the extension.
            import_module("visqol.pb2.similarity_result_pb2")
            self._status_error = import_module("pybind11_abseil.status").StatusNotOk
        except (ImportError, TypeError, RuntimeError) as exc:
            raise RuntimeError(f"Install the official ViSQOL Python API. See the README's ViSQOL setup. {exc}") from exc

        extension = Path(visqol_lib_py.__file__)
        model_name = Path(visqol_lib_py.DefaultSpeechModelFile()).name
        model_path = extension.parent / "model" / model_name
        self.model_digest = content_digest(model_path)
        self.versions = {name: version(name) for name in ("visqol", "protobuf")}
        self.parameters = {
            "sample_rate": EVALUATION_SAMPLE_RATE,
            "mode": "speech",
            "use_lattice_model": True,
            "search_window_radius": 60,
            "min_active_samples": MODEL_QUALITY_MIN_ACTIVE_SAMPLES,
            "max_span_samples": 9 * EVALUATION_SAMPLE_RATE,
            "edge_padding_samples": EVALUATION_SAMPLE_RATE // 2,
            "dtype": "float64",
            "extension_sha256": content_digest(extension),
        }
        config = visqol_config_pb2.VisqolConfig()
        config.audio.sample_rate = self.parameters["sample_rate"]
        config.options.use_speech_scoring = self.parameters["mode"] == "speech"
        config.options.use_lattice_model = self.parameters["use_lattice_model"]
        config.options.svr_model_path = str(model_path)
        config.options.search_window_radius = self.parameters["search_window_radius"]
        self._api = visqol_lib_py.VisqolApi()
        try:
            self._api.Create(config)
        except self._status_error as exc:
            raise RuntimeError(f"ViSQOL could not initialize its speech model: {exc}") from exc
        except TypeError as exc:
            raise RuntimeError(
                "ViSQOL could not read its configuration. See the README for compatible protobuf versions."
            ) from exc

    def __call__(self, reference: NDArray[np.float32], estimate: NDArray[np.float32]) -> MetricScore:
        """Score the same padded utterance prefix in both signals, reference first."""
        if min(reference.size, estimate.size) < self.parameters["min_active_samples"]:
            return MetricScore("ineligible", reason="visqol_buffer_too_short")
        # Trim both signals at the same sample and put identical silence around them.
        span = self.parameters["max_span_samples"]
        padding = self.parameters["edge_padding_samples"]
        dtype = self.parameters["dtype"]
        reference_prefix = np.ascontiguousarray(reference[:span], dtype=dtype)
        estimate_prefix = np.ascontiguousarray(estimate[:span], dtype=dtype)
        padded_reference = np.pad(reference_prefix, padding)
        padded_estimate = np.pad(estimate_prefix, padding)
        try:
            result = self._api.Measure(padded_reference, padded_estimate)
        except (self._status_error, RuntimeError, IndexError) as exc:
            return MetricScore("failed", reason=f"visqol_backend_error: {exc}")
        return finite_score(result.moslqo)


class DnsmosP808:
    """TorchMetrics DNSMOS-P.808 for nonempty mono float32 separated utterances.

    The pipeline requires three seconds of detected separated speech. TorchMetrics
    extends short inputs and averages their internal window scores. Native
    scoring failures and nonfinite P.808 scores are failed outcomes.
    """

    def __init__(self) -> None:
        """Fingerprint both local assets and preload their TorchMetrics CPU sessions."""
        import torch
        from onnxruntime.capi import onnxruntime_pybind11_state as ort
        from torchmetrics.functional.audio import dnsmos

        self._call = dnsmos.deep_noise_suppression_mean_opinion_score
        self._torch = torch
        self._scoring_errors = (RuntimeError, ort.Fail, ort.InvalidArgument, ort.RuntimeException, ort.EPFail)
        self.versions = {name: version(name) for name in ("torchmetrics", "onnxruntime", "requests")}
        self.parameters = {
            "sample_rate": EVALUATION_SAMPLE_RATE,
            "device": "cpu",
            "personalized": False,
            "cache_session": True,
            "component": "p808_mos",
            "window_seconds": dnsmos.INPUT_LENGTH,
            "min_active_samples": MODEL_QUALITY_MIN_ACTIVE_SAMPLES,
        }
        companion = "pDNSMOS" if self.parameters["personalized"] else "DNSMOS"
        model_paths = (
            f"{dnsmos.DNSMOS_DIR}/DNSMOS/model_v8.onnx",
            f"{dnsmos.DNSMOS_DIR}/{companion}/sig_bak_ovr.onnx",
        )
        try:
            self.model_digests = {
                f"dnsmos/{Path(path).name}": content_digest(Path(path).expanduser()) for path in model_paths
            }
        except OSError as exc:
            raise RuntimeError(f"DNSMOS requires both local ONNX models. See the README's DNSMOS setup. {exc}") from exc

        # Use the same session cache as the functional API to load each model once.
        # Refresh it for repeated invocations so a replacement model matches its digest.
        load_session = dnsmos._cached_load_session
        load_session.cache_clear()
        device = torch.device(self.parameters["device"])
        try:
            for path in model_paths:
                load_session(path, device, None)
        except (*self._scoring_errors, ort.InvalidProtobuf, ort.InvalidGraph, ort.NoModel, ort.NoSuchFile) as exc:
            raise RuntimeError(f"DNSMOS could not load its models. See the README's DNSMOS setup. {exc}") from exc

    def __call__(self, estimate: NDArray[np.float32]) -> MetricScore:
        """Score a complete separated utterance at 16 kHz and retain P.808 MOS."""
        if estimate.size < self.parameters["min_active_samples"]:
            return MetricScore("ineligible", reason="dnsmos_buffer_too_short")
        contiguous_estimate = np.ascontiguousarray(estimate)
        prediction = self._torch.from_numpy(contiguous_estimate)
        try:
            scores = self._call(
                prediction,
                self.parameters["sample_rate"],
                personalized=self.parameters["personalized"],
                device=self.parameters["device"],
                cache_session=self.parameters["cache_session"],
            )
        except self._scoring_errors as exc:
            return MetricScore("failed", reason=f"dnsmos_backend_error: {exc}")
        components = ("p808_mos", "mos_sig", "mos_bak", "mos_ovr")
        component_index = components.index(self.parameters["component"])
        return finite_score(scores[component_index])
