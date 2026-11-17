"""Evaluation discovery uses filenames and local case-insensitive filters."""

from pathlib import Path

import pytest

from realtime_speech_separation.eval_cli import get_inference_hash, get_input_files, get_separated_files, main


@pytest.fixture
def nested_audio(tmp_path: Path) -> Path:
    """Create nested variants, misleading companions, and sidecars."""
    names = (
        "DatasetA/session/call_name.wav",
        "DatasetA/session/call_name_reference.wav",
        "DatasetA/session/call_name_enrollment.wav",
        "DatasetA/session/call_name_separated_config_one.WAV",
        "DatasetA/session/call_name_separated_config_two_more.flac",
        "DatasetA/session/call_name_separated_other.mp3",
        "DatasetA/session/call_name_separated_other.opus",
        "DatasetA/session/call_name_separated_.wav",
        "DatasetA/session/call_name_separated_config_one.wav.json",
        "DatasetA/session/call_name_separated_config_one.txt",
        "DatasetA/session/call_name_separated_config_one.wav.bak",
        "DatasetA/session/call_name_results_config_one.json",
        "DatasetA/session/call_name_extra_separated_config_one.wav",
        "DatasetA/session/not_call_name_separated_config_one.wav",
        "DatasetB/session/other.FLAC",
        "DatasetB/session/other_reference.FLAC",
        "DatasetB/session/other_separated_config_one.wav",
        "DatasetA/session/nested/call_name_separated_not_adjacent.wav",
        "DatasetA/archive_reference.wav/call.wav",
        "DatasetA/archive_enrollment.wav/call.wav",
        "DatasetA/archive_separated_config/call.wav",
    )
    for name in reversed(names):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "DatasetA/session/call_name_separated_directory.wav").mkdir()
    return tmp_path


def test_nested_discovery_and_filters(nested_audio: Path) -> None:
    """Sort deterministically and apply OR filters only to mixture discovery."""
    root = nested_audio
    first = str(root / "DatasetA/session/call_name.wav")
    second = str(root / "DatasetB/session/other.FLAC")
    assert get_input_files(str(root)) == [first, second]
    assert get_input_files(str(root), "dataseta/SESSION") == [first]
    assert get_input_files(str(root), ["DATASETB", "call_NAME.wav"]) == [first, second]
    assert get_input_files(str(root), ["missing", "datasetb"]) == [second]
    assert get_input_files(str(root), "call_name_separated") == []
    assert get_input_files(str(root), "missing") == []
    # Filters keep their whitespace, matching the selection identity contract.
    assert get_input_files(str(root), " DatasetA") == []

    outputs = get_separated_files(first)
    assert outputs == [
        str(root / "DatasetA/session/call_name_separated_config_one.WAV"),
        str(root / "DatasetA/session/call_name_separated_config_two_more.flac"),
        str(root / "DatasetA/session/call_name_separated_other.mp3"),
        str(root / "DatasetA/session/call_name_separated_other.opus"),
    ]
    assert [get_inference_hash(first, output) for output in outputs] == [
        "config_one",
        "config_two_more",
        "other",
        "other",
    ]
    assert get_separated_files(second) == [str(root / "DatasetB/session/other_separated_config_one.wav")]


@pytest.mark.parametrize(
    "name", ["call_separated_.wav", "other_separated_config.wav", "call_more_separated_config.wav"]
)
def test_invalid_inference_suffix(name: str) -> None:
    """Only a nonempty suffix following the exact mixture stem is accepted."""
    with pytest.raises(ValueError, match="does not match"):
        get_inference_hash("call.wav", name)


def test_cli_help_without_optional_runtime_stacks(capsys: pytest.CaptureFixture[str]) -> None:
    """Help can describe triples before importing the legacy scoring backend."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "mixture/reference/separated triples" in capsys.readouterr().out


def test_cli_rejects_nonprotocol_sample_rate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A legacy flag cannot silently select a rate different from the loader."""
    with pytest.raises(SystemExit) as exc:
        main(["--input_dir", str(tmp_path), "--sample_rate", "8000"])
    assert exc.value.code == 2
    assert "requires --sample_rate=16000" in capsys.readouterr().err
