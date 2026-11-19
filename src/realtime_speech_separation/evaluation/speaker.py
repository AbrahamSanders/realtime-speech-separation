# Frontend adapted for in-memory audio from WeSpeaker's wespeaker/bin/infer_onnx.py.
# Copyright (c) 2022, Shuai Wang (wsstriving@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Provision the WeSpeaker model and extract reference and separated embeddings."""

import os
import shutil
import tempfile
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray

from .audio import EVALUATION_SAMPLE_RATE
from .reporting import content_digest
from .types import (
    IntervalResultStatus,
    ReferenceEmbeddingExtractionResult,
    ReferencePrototypeExtractionResult,
)
from .vad import ChannelSpeech

MODEL_NAME = "voxblink2_samresnet34.onnx"
MODEL_REVISION = "548e87a3fa9fb727a7d08d1387e1f710093c56dd"
MODEL_SHA256 = "1564bb8490c7f7759e293bbc2b2d326b5dc2b79f24e47a0d57f4a24064a58ad9"
MODEL_URL = f"https://modelscope.cn/datasets/wenet/wespeaker_pretrained_models/resolve/{MODEL_REVISION}/{MODEL_NAME}"


def cached_speaker_model() -> Path:
    """Fetch the pinned ONNX export on a cache miss and publish only verified bytes."""
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    path = cache_root / "rtss" / "wespeaker" / MODEL_REVISION / MODEL_NAME
    try:
        if path.is_file() and content_digest(path) == MODEL_SHA256:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)

        # Concurrent downloads can each finish safely without exposing partial files.
        stream = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{MODEL_NAME}.", delete=False)
        temporary = Path(stream.name)
        try:
            with stream:
                with urlopen(MODEL_URL, timeout=60) as response:
                    shutil.copyfileobj(response, stream)
            if content_digest(temporary) != MODEL_SHA256:
                raise RuntimeError("downloaded ONNX file did not match the expected SHA-256")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    except (OSError, HTTPException, RuntimeError) as exc:
        raise RuntimeError(
            f"Cannot cache WeSpeaker model at {path}: {exc}. Use --speaker_model_path for a local copy."
        ) from exc
    return path


@dataclass(frozen=True)
class ReferencePrototype:
    """A channel's reporting outcome and its normalized in-memory mean vector."""

    outcome: ReferencePrototypeExtractionResult
    unit_vector: NDArray[np.float64] | None = None


class WeSpeaker:
    """One offline CPU session for the multilingual VoxBlink2 SimAMResNet34 export."""

    def __init__(self, model_path: Path) -> None:
        """Fingerprint and validate the local variable-length ONNX model at preflight."""
        try:
            self.model_digest = content_digest(model_path)
        except OSError as exc:
            raise RuntimeError(f"Cannot read WeSpeaker model at {model_path}: {exc}") from exc

        try:
            import onnxruntime as ort
            import torch
            import torchaudio
            from onnxruntime.capi import onnxruntime_pybind11_state as ort_errors
            from torchaudio.compliance.kaldi import fbank
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"Install compatible runtimes with the eval extra: {exc}") from exc

        self._torch = torch
        self._fbank = fbank
        self._runtime_errors = (RuntimeError, ort_errors.Fail, ort_errors.InvalidArgument, ort_errors.RuntimeException)
        provider = "CPUExecutionProvider"
        model_errors = (
            *self._runtime_errors,
            ort_errors.InvalidGraph,
            ort_errors.InvalidProtobuf,
            ort_errors.NoSuchFile,
            ort_errors.NotImplemented,
        )
        try:
            self._session = ort.InferenceSession(str(model_path), providers=[provider])
        except model_errors as exc:
            raise RuntimeError(f"Cannot load WeSpeaker model at {model_path}: {exc}") from exc

        # Require the official export's interface, including its dynamic time axis.
        inputs, outputs = self._session.get_inputs(), self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"Expected one input and one output in WeSpeaker model at {model_path}")
        (features,), (embedding,) = inputs, outputs
        if (
            features.name != "feats"
            or features.type != "tensor(float)"
            or len(features.shape) != 3
            or (isinstance(features.shape[0], int) and features.shape[0] != 1)
            or isinstance(features.shape[1], int)
            or features.shape[2] != 80
            or embedding.name != "embs"
            or embedding.type != "tensor(float)"
            or len(embedding.shape) != 2
            or (isinstance(embedding.shape[0], int) and embedding.shape[0] != 1)
            or embedding.shape[1] != 256
        ):
            raise RuntimeError(f"Expected the variable-length {MODEL_NAME} export at {model_path}")

        self.backend_version = f"WeSpeaker ONNX via ONNX Runtime {ort.__version__}"
        self.versions = {
            "torch": str(torch.__version__),
            "torchaudio": torchaudio.__version__,
            "onnxruntime": ort.__version__,
        }
        self.parameters = {
            "model": MODEL_NAME,
            "provider": provider,
            "input_name": features.name,
            "output_name": embedding.name,
            "embedding_dimensions": embedding.shape[1],
            "waveform_scale": 1 << 15,
            "fbank": {
                "sample_frequency": EVALUATION_SAMPLE_RATE,
                "num_mel_bins": features.shape[2],
                "frame_length": 25,
                "frame_shift": 10,
                "dither": 0.0,
                "window_type": "hamming",
                "use_energy": False,
            },
            "feature_normalization": "utterance_mean",
            "min_active_samples": 3 * EVALUATION_SAMPLE_RATE,
            "reference_average": "equal_weight_raw_embeddings",
            "score": "raw_cosine",
        }

    def __call__(
        self, samples: NDArray[np.float32]
    ) -> tuple[NDArray[np.float64] | None, IntervalResultStatus, str | None]:
        """Return a raw (256,) vector and extraction outcome for a 16 kHz mono utterance."""
        parameters = self.parameters
        try:
            with self._torch.inference_mode():
                waveform = self._torch.as_tensor(samples, device="cpu").unsqueeze(0)
                scaled_waveform = waveform * parameters["waveform_scale"]
                features = self._fbank(scaled_waveform, **parameters["fbank"])
                centered_features = features - features.mean(dim=0)
                batch = centered_features.unsqueeze(0).numpy()
            (embedding,) = self._session.run([parameters["output_name"]], {parameters["input_name"]: batch})
        except self._runtime_errors as exc:
            return None, "failed", f"speaker_backend_error: {exc}"
        if embedding.shape != (1, parameters["embedding_dimensions"]):
            return None, "failed", f"speaker_backend_error: unexpected embedding shape {embedding.shape}"
        vector = embedding[0].astype(np.float64)
        if not np.isfinite(vector).all():
            return None, "failed", "nonfinite_embedding"
        if not np.any(vector):
            return None, "undefined", "zero_norm_embedding"
        return vector, "scored", None

    def build_reference_prototype(
        self, samples: list[NDArray[np.float32]], speech: ChannelSpeech
    ) -> ReferencePrototype:
        """Average eligible embeddings for one reference channel, retaining extraction outcomes.

        Supply one mono 16 kHz waveform per region in `speech.utterances`, in
        the same order. Regions provide recording bounds and detected speech duration.
        """
        if speech.error is not None:
            outcome = ReferencePrototypeExtractionResult("failed", reason="reference_vad_error")
            return ReferencePrototype(outcome)

        min_active_samples = self.parameters["min_active_samples"]
        embeddings, extractions = [], []
        for waveform, region in zip(samples, speech.utterances, strict=True):
            interval = region.interval
            if region.active_samples < min_active_samples:
                extractions.append(ReferenceEmbeddingExtractionResult(interval, "ineligible", "insufficient_activity"))
                continue
            embedding, status, reason = self(waveform)
            extractions.append(ReferenceEmbeddingExtractionResult(interval, status, reason))
            if embedding is not None:
                embeddings.append(embedding)

        # Every eligible reference utterance must supply a usable vector to the mean.
        outcomes = tuple(extractions)
        if any(item.status == "failed" for item in outcomes):
            outcome = ReferencePrototypeExtractionResult("failed", outcomes, "reference_embedding_error")
            return ReferencePrototype(outcome)
        if any(item.status == "undefined" for item in outcomes):
            outcome = ReferencePrototypeExtractionResult("unavailable", outcomes, "zero_norm_reference_embedding")
            return ReferencePrototype(outcome)
        if not embeddings:
            outcome = ReferencePrototypeExtractionResult("unavailable", outcomes, "no_eligible_reference_utterances")
            return ReferencePrototype(outcome)
        mean_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm == 0:
            outcome = ReferencePrototypeExtractionResult("unavailable", outcomes, "zero_norm_prototype")
            return ReferencePrototype(outcome)
        outcome = ReferencePrototypeExtractionResult("available", outcomes)
        return ReferencePrototype(outcome, mean_embedding / norm)
