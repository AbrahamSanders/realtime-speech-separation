"""Installation smoke tests."""

import json
import subprocess
import sys
from importlib.metadata import distribution, metadata
from pathlib import Path

import realtime_speech_separation


def test_installed_metadata() -> None:
    distribution = metadata("realtime-speech-separation")

    assert distribution["Name"] == "realtime-speech-separation"
    assert realtime_speech_separation.__version__ == distribution["Version"]


def test_import_outside_repository(tmp_path: Path) -> None:
    # Isolated mode ignores PYTHONPATH and the current directory, exercising
    # the installation rather than accidentally importing the source tree.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json, sys, realtime_speech_separation; "
            "assert not {'torch', 'transformers', 'codec_bpe', 'llama_cpp', "
            "'asteroid', 'gradio'} & sys.modules.keys(); "
            "print(json.dumps(realtime_speech_separation.__version__))",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert json.loads(result.stdout) == realtime_speech_separation.__version__


def test_installed_console_scripts() -> None:
    """Ensure dotted CLI names are registered as console entry points."""
    entry_points = {
        entry.name: entry.value
        for entry in distribution("realtime-speech-separation").entry_points
        if entry.group == "console_scripts"
    }

    assert entry_points == {
        "rtss.separate": "realtime_speech_separation.inference_cli:main",
        "rtss.client": "realtime_speech_separation.inference_client:main",
        "rtss.eval": "realtime_speech_separation.eval_cli:main",
    }
