"""Realtime speech separation, with optional runtime components loaded on demand."""

from importlib import import_module
from importlib.metadata import version
from typing import Any

__version__ = version("realtime-speech-separation")

_EXPORT_MODULES = {
    "AudioTokenizer": ".audio_tokenizer",
    "RealtimeSeparatorResources": ".realtime_separator_resources",
    "RealtimeSeparatorConfig": ".realtime_separator_config",
    "RealtimeSeparator": ".realtime_separator",
    "RealtimeSeparatorMultiprocessing": ".realtime_separator",
    "add_common_inference_args": ".utils.cli_utils",
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Load public components when accessed, leaving optional stacks unloaded."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value
