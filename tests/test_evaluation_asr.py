"""Exercise the ASR adapter with a fake binding and opt-in native transcription."""

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from realtime_speech_separation.evaluation import asr
from realtime_speech_separation.evaluation.types import Transcript, WordCounts


@pytest.fixture
def whisper_runtime(tmp_path, monkeypatch):
    """Inject fake optional modules and an asset with a matching checksum."""
    path = tmp_path / asr.MODEL_NAME
    path.write_bytes(b"test-model")
    monkeypatch.setattr(asr, "MODEL_SHA256", hashlib.sha256(b"test-model").hexdigest())
    monkeypatch.setattr(asr, "version", lambda name: f"test-{name}")
    state = SimpleNamespace(path=path, detections=[], decodes=[], edits=[], probability=0.8)

    class Model:
        """Simulate the binding's persistent decode parameters and stripped segments."""

        def __init__(self, model, **options):
            """Record the model path and CPU settings."""
            state.loaded = (model, options)
            self.settings = {}

        @staticmethod
        def available_languages():
            """Offer two recognizable language codes."""
            return ["en", "fr"]

        def auto_detect_language(self, waveform, offset_ms):
            """Report a reference language from the waveform marker."""
            assert waveform.ndim == 1 and waveform.dtype == np.float32 and waveform.flags.c_contiguous
            state.detections.append((waveform.copy(), offset_ms))
            language = "en" if waveform[0] == 0 else "fr"
            return (language, np.float32(state.probability)), {language: state.probability}

        def transcribe(self, waveform, **settings):
            """Retain overrides between calls and return two segments."""
            assert waveform.ndim == 1 and waveform.dtype == np.float32 and waveform.flags.c_contiguous
            self.settings.update(settings)
            state.decodes.append((waveform.copy(), self.settings.copy()))
            self.settings.update(language="stale", no_context=False, initial_prompt="stale", prompt_tokens=[5])
            return [SimpleNamespace(text="One!"), SimpleNamespace(text="Two.")]

    def process_words(reference, separated):
        """Capture normalized text and return deterministic edit counts."""
        state.edits.append((reference, separated))
        return SimpleNamespace(hits=2, substitutions=0, deletions=0, insertions=0)

    monkeypatch.setitem(sys.modules, "pywhispercpp.model", SimpleNamespace(Model=Model))
    monkeypatch.setitem(sys.modules, "jiwer", SimpleNamespace(process_words=process_words))
    return state


def test_adapter_uses_full_contiguous_channels_and_resets_decode_settings(whisper_runtime):
    """Use only reference detections, reapply decode settings, and keep raw segment text."""
    state = whisper_runtime
    adapter = asr.WhisperWer(state.path, None)
    assert state.loaded == (
        str(state.path),
        {"params_sampling_strategy": 0, "context_params": {"use_gpu": False, "flash_attn": False}},
    )
    channels = np.arange(32000, dtype=np.float64).reshape(-1, 2)
    for channel in (0, 1):
        reference = adapter.transcribe_reference(channels[:, channel])
        score = adapter.compare(reference, channels[:, 1 - channel])
        assert score.counts == WordCounts(0, 0, 0, 2)
        assert score.separated == reference
    for index, (waveform, offset) in enumerate(state.detections):
        np.testing.assert_array_equal(waveform, channels[:, index])
        assert offset == 0
    assert len(state.detections) == 2
    expected = [(0, "en"), (1, "en"), (1, "fr"), (0, "fr")]
    for (waveform, settings), (channel, language) in zip(state.decodes, expected, strict=True):
        np.testing.assert_array_equal(waveform, channels[:, channel])
        assert settings == {"language": language, **asr.DECODE_OPTIONS}
    assert state.edits == [("one two", "one two")] * 2
    assert reference == Transcript("One! Two.", "one two", "fr")


def test_override_and_invalid_detection(whisper_runtime):
    """Validate overrides before decoding and reject a silent native detection failure."""
    state = whisper_runtime
    waveform = np.zeros(16000, dtype=np.float32)
    adapter = asr.WhisperWer(state.path, " FR ")
    reference = adapter.transcribe_reference(waveform)
    assert adapter.compare(reference, waveform).separated.language == "fr"
    assert adapter.parameters["language_override"] == "fr"
    assert not state.detections
    with pytest.raises(RuntimeError, match="Unsupported --asr_language"):
        asr.WhisperWer(state.path, "invalid")
    state.probability = 0.0
    with pytest.raises(asr.AsrBackendError, match="language_detection_error"):
        asr.WhisperWer(state.path, None).transcribe_reference(waveform)


def test_transcript_normalization():
    """Apply NFC and lowercase, delete Unicode punctuation, and preserve symbols."""
    assert asr.normalize_transcript("  CAFE\u0301 — CAN’T、\tA-B\nＹ！ + 💡\u00a0例。 ") == "café cant ab ｙ + 💡 例"


def test_jiwer_counts_known_pairs(monkeypatch):
    """Check JiWER 4 edit counts with ordinary and empty transcripts when installed."""
    jiwer = pytest.importorskip("jiwer", exc_type=ImportError)
    adapter = asr.WhisperWer.__new__(asr.WhisperWer)
    adapter._process_words = jiwer.process_words
    waveform = np.zeros(16000, dtype=np.float32)
    for reference, hypothesis, expected in (
        ("one. A B C", "one X B", WordCounts(1, 1, 0, 4)),
        ("", "One two.", WordCounts(0, 0, 2, 0)),
        ("one two", "", WordCounts(0, 2, 0, 2)),
        ("", "", WordCounts(0, 0, 0, 0)),
    ):
        transcript = Transcript(hypothesis, asr.normalize_transcript(hypothesis), "en")
        monkeypatch.setattr(adapter, "_transcribe", lambda *_, transcript=transcript: transcript)
        result = adapter.compare(Transcript(reference, asr.normalize_transcript(reference), "en"), waveform)
        assert result.status == "scored"
        assert result.counts == expected


@pytest.mark.metric_backend
@pytest.mark.skipif(os.environ.get("RTSS_TEST_ASR") != "1", reason="requires local large-v3-turbo and JFK audio")
def test_native_large_v3_turbo():
    """Verify the released binding's CPU language detection and repeatable speech decode."""
    assets = Path(os.environ["RTSS_ASR_TEST_ASSETS"])
    waveform, rate = sf.read(assets / "jfk.wav", dtype="float32")
    assert rate == 16000
    adapter = asr.WhisperWer(assets / asr.MODEL_NAME, None)
    reference = adapter.transcribe_reference(waveform)
    assert reference.language == "en"
    assert "what your country can do for you" in reference.normalized
    result = adapter.compare(reference, waveform)
    assert result.separated == reference
    assert result.counts == WordCounts(0, 0, 0, len(reference.normalized.split()))
