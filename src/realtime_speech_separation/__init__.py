"""Realtime Speech Separation."""

from importlib.metadata import version

__version__ = version("speech-separation-eval")

from .audio_tokenizer import AudioTokenizer
from .realtime_separator_resources import RealtimeSeparatorResources
from .realtime_separator_config import RealtimeSeparatorConfig
from .realtime_separator import RealtimeSeparator, RealtimeSeparatorMultiprocessing

from .utils.cli_utils import add_common_inference_args