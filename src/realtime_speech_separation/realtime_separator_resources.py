import torch
import os
from typing import Union, Optional
from transformers import AutoTokenizer

from . import AudioTokenizer
from .utils.llamacpp_utils import LlamaForAlternatingCodeChannels

class RealtimeSeparatorResources:
    def __init__(
        self, 
        llm_model_path: str = "Qwen3-1.7B-magicodec-no-bpe-131k-test/Qwen3-1.7B-magicodec-no-bpe-131k-test-F16.gguf", 
        llm_n_ctx: int = 16384,
        codec_model: str = "MagiCodec-50Hz-Base", 
        codec_device: Optional[Union[str, torch.device]] = None,
    ):
        self.llm_model_dir = os.path.dirname(llm_model_path)
        self.llm = LlamaForAlternatingCodeChannels(
            model_path=llm_model_path,
            n_ctx=llm_n_ctx,
            n_gpu_layers=-1,
            verbose=False,
            flash_attn=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_dir)
        if codec_device is None:
            codec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mono_audio_tokenizer = AudioTokenizer(codec_model=codec_model, device=codec_device)
        self.stereo_audio_tokenizer = AudioTokenizer(codec_model=self.mono_audio_tokenizer.codec_model, num_channels=2, device=codec_device)
