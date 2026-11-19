"""Transcribe full evaluation channels with Whisper and count reference ASR edits."""

import os
import unicodedata
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .reporting import content_digest
from .types import Transcript, WordCounts, WordErrorResult

MODEL_NAME = "ggml-large-v3-turbo.bin"
MODEL_SHA256 = "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"
LANGUAGE_DETECTION_OFFSET_MS = 0
MODEL_OPTIONS = {"params_sampling_strategy": 0, "context_params": {"use_gpu": False, "flash_attn": False}}
DECODE_OPTIONS = {
    "offset_ms": 0,
    "duration_ms": 0,
    "translate": False,
    "detect_language": False,
    "no_context": True,
    "initial_prompt": "",
    "prompt_tokens": None,
    "carry_initial_prompt": False,
    "greedy": {"best_of": 1},
    "temperature": 0.0,
    "temperature_inc": 0.0,
    "vad": False,
    "print_progress": False,
}


class AsrBackendError(RuntimeError):
    """A channel's language detection or transcription failed."""


def normalize_transcript(text: str) -> str:
    """Apply protocol NFC, lowercase, punctuation deletion, and whitespace folding."""
    lowered = unicodedata.normalize("NFC", text).lower()
    without_punctuation = "".join(char for char in lowered if not unicodedata.category(char).startswith("P"))
    return " ".join(without_punctuation.split())


class WhisperWer:
    """Reuse one CPU large-v3-turbo model within a sequential evaluation run."""

    def __init__(self, model_path: Path | None, language: str | None) -> None:
        """Check the local F16 export and language override, then load the optional backends."""
        if model_path is None:
            cache_home = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
            model_path = Path(cache_home) / "rtss" / "asr" / MODEL_NAME
        model_path = model_path.expanduser().resolve()
        if not model_path.is_file():
            raise RuntimeError(f"Missing Whisper model at {model_path}. See README Whisper setup or --asr_model_path.")
        self.model_digest = content_digest(model_path)
        if self.model_digest != MODEL_SHA256:
            raise RuntimeError(f"Unexpected Whisper model checksum at {model_path}. See README Whisper setup.")

        try:
            from jiwer import process_words
            from pywhispercpp.model import Model
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"Install the Whisper and JiWER runtimes with the eval extra: {exc}") from exc

        self._languages = frozenset(Model.available_languages())
        self._language = language.strip().lower() if language is not None else None
        if self._language is not None and self._language not in self._languages:
            raise RuntimeError(f"Unsupported --asr_language code: {language!r}")
        self._model = Model(str(model_path), **MODEL_OPTIONS)
        self._process_words = process_words
        self.backend_version = version("pywhispercpp")
        self.versions = {"jiwer": version("jiwer")}
        self.parameters = {
            "recognizer": "pywhispercpp.model.Model",
            "model": MODEL_NAME,
            "quantization": "f16",
            "model_options": MODEL_OPTIONS,
            "decode_options": DECODE_OPTIONS,
            "language_policy": "reference_channel",
            "language_override": self._language,
            "language_detection_offset_ms": LANGUAGE_DETECTION_OFFSET_MS,
            "normalizer": "nfc_lowercase_delete_unicode_punctuation_whitespace_v1",
            "unicode_version": unicodedata.unidata_version,
            "comparison": "agreement_with_reference_asr",
        }

    def transcribe_reference(self, samples: NDArray[np.float32]) -> Transcript:
        """Detect language from the full 16 kHz mono reference, then transcribe it."""
        waveform = np.ascontiguousarray(samples, dtype=np.float32)
        language = self._language
        if language is None:
            try:
                (language, probability), _ = self._model.auto_detect_language(
                    waveform, offset_ms=LANGUAGE_DETECTION_OFFSET_MS
                )
            except RuntimeError as exc:
                raise AsrBackendError(f"language_detection_error: {exc}") from exc
            # In 1.5.1 a negative native status can select a Python list entry with zero probability.
            if language not in self._languages or not 0 < probability <= 1:
                raise AsrBackendError("language_detection_error: Whisper returned no valid language probability")
        return self._transcribe(waveform, language)

    def _transcribe(self, waveform: NDArray[np.float32], language: str) -> Transcript:
        """Transcribe one contiguous 16 kHz mono float32 channel with fresh settings."""
        # Model.transcribe retains overrides between calls, so reapply the language and prompt settings.
        try:
            segments = self._model.transcribe(waveform, language=language, **DECODE_OPTIONS)
        except RuntimeError as exc:
            raise AsrBackendError(f"transcription_error: {exc}") from exc
        raw = " ".join(segment.text for segment in segments)
        normalized = normalize_transcript(raw)
        return Transcript(raw, normalized, language)

    def compare(self, reference: Transcript, samples: NDArray[np.float32]) -> WordErrorResult:
        """Transcribe a full 16 kHz mono estimate in the reference language and count edits."""
        assert reference.language is not None
        waveform = np.ascontiguousarray(samples, dtype=np.float32)
        try:
            separated = self._transcribe(waveform, reference.language)
        except AsrBackendError as exc:
            return WordErrorResult("failed", reference=reference, reason=f"separated_{exc}")
        try:
            edits = self._process_words(reference.normalized, separated.normalized)
        except ValueError as exc:
            return WordErrorResult(
                "failed", reference=reference, separated=separated, reason=f"word_alignment_error: {exc}"
            )
        reference_words = edits.hits + edits.substitutions + edits.deletions
        counts = WordCounts(edits.substitutions, edits.deletions, edits.insertions, reference_words)
        return WordErrorResult("scored", counts, reference, separated)
