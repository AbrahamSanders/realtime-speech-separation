"""Exercise the CLI's audio boundary without optional scoring dependencies."""

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from realtime_speech_separation.eval_cli import main


@pytest.fixture
def legacy_metrics(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Stub legacy scoring/reporting until the protocol pipeline replaces them."""
    metrics = ModuleType("asteroid.metrics")
    scorer = Mock()
    metrics.get_metrics = scorer
    monkeypatch.setitem(sys.modules, "asteroid", ModuleType("asteroid"))
    monkeypatch.setitem(sys.modules, "asteroid.metrics", metrics)
    pandas = ModuleType("pandas")
    pandas.DataFrame = Mock()
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    return scorer


@pytest.mark.parametrize("channels", [1, 2])
def test_cli_loads_triple_without_enrollment(tmp_path: Path, legacy_metrics: Mock, channels: int) -> None:
    """The CLI passes validated T or T/R audio to its current metric backend."""
    mixture = np.linspace(-0.5, 0.5, 128, dtype=np.float32)
    reference = np.stack((mixture, -mixture), axis=1)
    separated = reference[:, :channels]
    for name, samples in (
        ("call_name.wav", mixture),
        ("call_name_reference.wav", reference),
        ("call_name_separated_config_with_suffix.wav", separated),
    ):
        sf.write(tmp_path / name, samples, 16000, subtype="FLOAT")
    legacy_metrics.return_value = {"si_sdr": np.zeros(channels)}

    assert main(["--input_dir", str(tmp_path), "--input_filter", "CALL_NAME.wav", "--overwrite"]) == 0

    legacy_metrics.assert_called_once()
    arguments = legacy_metrics.call_args.kwargs
    np.testing.assert_array_equal(arguments["mix"], mixture)
    np.testing.assert_array_equal(arguments["clean"], reference[:, :channels].T)
    np.testing.assert_array_equal(arguments["estimate"], separated.T)
    assert arguments["sample_rate"] == 16000
    assert not (tmp_path / "call_name_enrollment.wav").exists()
    assert (tmp_path / "call_name_results_config_with_suffix.json").is_file()
    summary = json.loads((tmp_path / "summary_config_with_suffix.json").read_text())
    assert summary["total_count"] == 1
    assert len(summary["si_sdr"]["mean"]) == channels


def test_cli_continues_after_invalid_audio(
    tmp_path: Path, legacy_metrics: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    """A bad triple does not block a valid one, but makes the run unsuccessful."""
    mixture = np.zeros(128, dtype=np.float32)
    reference = np.zeros((128, 2), dtype=np.float32)
    for name, samples in (
        ("a_invalid.wav", mixture),
        ("a_invalid_separated_hash.wav", reference),
        ("b_valid.wav", mixture),
        ("b_valid_reference.wav", reference),
        ("b_valid_separated_hash.wav", reference),
    ):
        sf.write(tmp_path / name, samples, 16000, subtype="FLOAT")
    legacy_metrics.return_value = {"si_sdr": np.zeros(2)}

    assert main(["--input_dir", str(tmp_path), "--overwrite"]) == 1

    legacy_metrics.assert_called_once()
    assert "a_invalid_reference.wav" in caplog.text
    assert not (tmp_path / "a_invalid_results_hash.json").exists()
    summary = json.loads((tmp_path / "summary_hash.json").read_text())
    assert summary["total_count"] == 1
