"""Lazy CPU metric adapters for validated, aligned interval waveforms."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
from numpy.typing import NDArray

from .audio import EVALUATION_SAMPLE_RATE
from .types import IntervalResultStatus


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
        self.parameters = {
            "zero_mean": True,
            "dtype": "float32",
            "device": "cpu",
            "epsilon": float(np.finfo(np.float32).eps),
            "epsilon_placement": "projection_numerator_denominator_and_both_energy_sums",
            "centered_zero_energy": "exact_float64_centering_without_positive_cutoff",
        }

    def __call__(self, reference: NDArray[np.float32], estimate: NDArray[np.float32]) -> MetricScore:
        """Preserve undefined silence rather than accepting epsilon placeholders."""
        if centered_zero_energy(reference):
            return MetricScore("ineligible", reason="zero_energy_reference")
        if centered_zero_energy(estimate):
            return MetricScore("undefined", reason="zero_energy_estimate")
        contiguous_reference = np.ascontiguousarray(reference)
        contiguous_estimate = np.ascontiguousarray(estimate)
        target = self._torch.from_numpy(contiguous_reference)
        prediction = self._torch.from_numpy(contiguous_estimate)
        try:
            score = self._call(prediction, target, zero_mean=True)
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
                score = self._call(reference, estimate, EVALUATION_SAMPLE_RATE, extended=False)
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

    def __init__(self, call: Callable | None = None) -> None:
        """Resolve the PESQ function and typed errors; allow injected calls in tests."""
        import pesq

        if call is None:
            call = pesq.pesq
        self._call = call
        self._backend = pesq
        self.versions = {"pesq": version("pesq")}
        self.parameters = {"sample_rate": EVALUATION_SAMPLE_RATE, "mode": "wb", "on_error": "raise_exception"}

    def __call__(self, reference: NDArray[np.float32], estimate: NDArray[np.float32]) -> MetricScore:
        """Score a reference utterance using the wrapper's typed error mode."""
        backend = self._backend
        try:
            score = self._call(
                EVALUATION_SAMPLE_RATE, reference, estimate, mode="wb", on_error=backend.PesqError.RAISE_EXCEPTION
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
