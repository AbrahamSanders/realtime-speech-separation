"""Installation smoke tests."""

import json
import subprocess
import sys
from importlib.metadata import metadata
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
            "import json, realtime_speech_separation; "
            "print(json.dumps(realtime_speech_separation.__version__))",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert json.loads(result.stdout) == realtime_speech_separation.__version__
