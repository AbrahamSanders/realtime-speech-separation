import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Model, Qwen3ForCausalLM, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.activations import ACT2FN
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack

class CodecQwen3Config(Qwen3Config):
    def __init__(
        self,
        num_codebooks=1,
        codebook_size=131072,
        codebook_dim=16,
        projector_hidden_act="gelu",
        codec_vocab_start=0,
        **kwargs
    ):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.projector_hidden_act = projector_hidden_act
        self.codec_vocab_start = codec_vocab_start
        super().__init__(**kwargs)

# Adapted from transformers.models.llava.modeling_llava.LlavaMultiModalProjector
class CodecQwen3MultiModalProjector(nn.Module):
    def __init__(self, config: CodecQwen3Config):
        super().__init__()

        self.linear_1 = nn.Linear(config.codebook_dim, config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, codec_token_embeds):
        hidden_states = self.linear_1(codec_token_embeds)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class CodecQwen3CodecEmbedding(nn.Module):
    def __init__(self, config: CodecQwen3Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.codec_embed = nn.Embedding(
            num_embeddings=config.num_codebooks * config.codebook_size,
            embedding_dim=config.codebook_dim,
            padding_idx=self.padding_idx,
            _freeze=True,  # Freeze the codec embeddings - it is the projector weights that will be trained!!!
        )
        self.codebook_projectors = nn.ModuleList(
            [CodecQwen3MultiModalProjector(config) for _ in range(config.num_codebooks)]
        )

    def forward(self, codec_input_ids: torch.Tensor) -> torch.Tensor:
        codec_input_ids = codec_input_ids - self.config.codec_vocab_start
        embeds = self.codec_embed(codec_input_ids)
        proj_embeds = torch.empty(codec_input_ids.shape + (self.config.hidden_size,), dtype=embeds.dtype, device=embeds.device)
        cb_size = self.config.codebook_size
        for i, codebook_proj in enumerate(self.codebook_projectors):
            codebook_tokens = (codec_input_ids >= i*cb_size) & (codec_input_ids < (i+1)*cb_size)
            proj_embeds[codebook_tokens] = codebook_proj(embeds[codebook_tokens]).to(proj_embeds.dtype)
        return proj_embeds

# Adapted from https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3/modeling_qwen3.py
@auto_docstring
class CodecQwen3Model(Qwen3Model):
    def __init__(self, config: CodecQwen3Config):
        super(Qwen3Model, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_codec_tokens = CodecQwen3CodecEmbedding(config)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = torch.empty(input_ids.shape + (self.config.hidden_size,), dtype=self.dtype, device=self.device)
            vocab_tokens = input_ids < self.config.codec_vocab_start
            codec_tokens = input_ids >= self.config.codec_vocab_start
            inputs_embeds[vocab_tokens] = self.embed_tokens(input_ids[vocab_tokens])
            inputs_embeds[codec_tokens] = self.embed_codec_tokens(input_ids[codec_tokens])

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
class CodecQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: CodecQwen3Config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = CodecQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_codec_embeddings(self, codec_embed_weight: torch.Tensor):
        assert (
            codec_embed_weight.shape == self.model.embed_codec_tokens.codec_embed.weight.shape and
            codec_embed_weight.dtype == self.model.embed_codec_tokens.codec_embed.weight.dtype
        ), (f"codec_embed_weight must be a {self.model.embed_codec_tokens.codec_embed.weight.dtype} tensor "
            f"of shape {self.model.embed_codec_tokens.codec_embed.weight.shape}")
        
        codec_embed_weight = codec_embed_weight.clone(memory_format=torch.contiguous_format).to(
            self.model.embed_codec_tokens.codec_embed.weight.device
        )
        self.model.embed_codec_tokens.codec_embed.weight.data = codec_embed_weight

    def persist_codec_embeddings(self, batch_size: int = 1024, show_progress: bool = True):
        # first we have to untie the embeddings from the LM head if they are tied, otherwise
        # we end up lobotomizing the region of the LM head that corresponds to the codec tokens!
        if getattr(self.config.get_text_config(decoder=True), "tie_word_embeddings"):
            setattr(self.config.get_text_config(decoder=True), "tie_word_embeddings", False)
            self._tied_weights_keys = []
            self.lm_head.weight = torch.nn.Parameter(self.lm_head.weight.clone())

        # now, project each codebook vector and save it in self.embed_tokens
        num_embeddings = self.config.num_codebooks * self.config.codebook_size
        codec_input_ids = torch.arange(
            self.config.codec_vocab_start, 
            self.config.codec_vocab_start + num_embeddings, 
            device=self.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            if show_progress:
                from tqdm import trange
                range_iter = trange(0, num_embeddings, batch_size, desc="Persisting codec embeddings")
            else:
                range_iter = range(0, num_embeddings, batch_size)
            for start in range_iter:
                end = start + batch_size
                batch_codec_input_ids = codec_input_ids[start:end]
                proj_embeds = self.model.embed_codec_tokens(batch_codec_input_ids)
                self.model.embed_tokens.weight.data[batch_codec_input_ids] = proj_embeds
                # sanity check
                assert torch.equal(self.model.embed_tokens(batch_codec_input_ids), proj_embeds), "proj_embeds does not match embed_tokens"