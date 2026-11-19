"""WeSpeaker cache tests and opt-in parity against the official CPU frontend."""

import importlib.util
import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from realtime_speech_separation.evaluation import speaker
from realtime_speech_separation.evaluation.speaker import MODEL_NAME, WeSpeaker


def test_speaker_cache_downloads_reuses_and_repairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A verified model works offline and a corrupt cache entry is replaced."""
    model = b"test model"
    requests = []

    def download(url: str, *, timeout: int) -> BytesIO:
        """Serve deterministic model bytes for each network request."""
        requests.append(url)
        return BytesIO(model)

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(speaker, "MODEL_SHA256", sha256(model).hexdigest())
    monkeypatch.setattr(speaker, "urlopen", download)
    path = speaker.cached_speaker_model()
    assert path == tmp_path / "rtss" / "wespeaker" / speaker.MODEL_REVISION / MODEL_NAME
    assert path.read_bytes() == model
    assert speaker.cached_speaker_model() == path
    assert requests == [speaker.MODEL_URL]
    path.write_bytes(b"corrupt cache")
    assert speaker.cached_speaker_model() == path
    assert path.read_bytes() == model
    assert requests == [speaker.MODEL_URL, speaker.MODEL_URL]


def test_speaker_cache_rejects_unverified_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checksum failures leave no model or temporary files in the cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(speaker, "urlopen", lambda url, timeout: BytesIO(b"wrong model"))
    with pytest.raises(RuntimeError, match="did not match the expected SHA-256"):
        speaker.cached_speaker_model()
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


@pytest.mark.metric_backend
@pytest.mark.skipif(os.environ.get("RTSS_TEST_WESPEAKER") != "1", reason="requires local WeSpeaker assets")
def test_wespeaker_matches_official_onnx_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare raw embeddings at distinct lengths using one model and the official frontend."""
    import onnxruntime as ort
    import torch

    assets = Path(os.environ["RTSS_WESPEAKER_TEST_ASSETS"])
    model = assets / MODEL_NAME
    example = importlib.util.spec_from_file_location("wespeaker_infer_onnx", assets / "infer_onnx.py")
    assert example is not None and example.loader is not None
    official = importlib.util.module_from_spec(example)
    example.loader.exec_module(official)

    def load_wav(path: Path) -> tuple[torch.Tensor, int]:
        """Use the same float WAV decoder for both paths, without TorchCodec."""
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio.T)
        return waveform, rate

    monkeypatch.setattr(official.torchaudio, "load", load_wav)
    extractor = WeSpeaker(model)
    options = ort.SessionOptions()
    options.inter_op_num_threads = options.intra_op_num_threads = 1
    official_session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])

    random = np.random.default_rng(812)
    for sample_count in (48000, 64321):
        waveform = random.uniform(-0.25, 0.25, size=sample_count).astype(np.float32)
        path = tmp_path / f"input_{sample_count}.wav"
        sf.write(path, waveform, 16000, subtype="FLOAT")
        features = official.compute_fbank(path).unsqueeze(0).numpy()
        (expected,) = official_session.run(["embs"], {"feats": features})
        actual, status, reason = extractor(waveform)
        assert (status, reason) == ("scored", None)
        assert actual is not None
        assert actual.shape == (256,)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected[0], rtol=1e-5, atol=1e-5)

    for value, outcome in ((0.0, ("undefined", "zero_norm_embedding")), (np.nan, ("failed", "nonfinite_embedding"))):
        invalid = np.full((1, 256), value, dtype=np.float32)
        monkeypatch.setattr(extractor._session, "run", lambda *_args, invalid=invalid: [invalid])
        assert extractor(waveform) == (None, *outcome)
