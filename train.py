#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "albumentations >= 1.4.16",
#     "accelerate >= 0.12.0",
#     "torch >= 1.3",
#     "datasets >= 2.14.0",
#     "sentencepiece != 0.1.92",
#     "protobuf",
#     "evaluate",
#     "scikit-learn",
#     "codec_bpe",
# ]
# ///

"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

# -------------------------------------------------------------------------------------------------------------------
# Adapted from https://github.com/huggingface/transformers/blob/v5.17.0/examples/pytorch/language-modeling/run_clm.py
# Modified for use with a line-by-line text file dataset containing tokenized audio.
# -------------------------------------------------------------------------------------------------------------------

import logging
import math
import os
import sys
from dataclasses import dataclass, field

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_xla_available,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from realtime_speech_separation.utils.training_utils import DataCollatorWithLossMasking
from realtime_speech_separation.codec_qwen3 import CodecQwen3ForCausalLM, CodecQwen3Config
from codec_bpe import UNICODE_OFFSET_LARGE

# torch.set_float32_matmul_precision('high')

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


def get_model_and_config_class(model_name: str, use_codec_proj: bool):
    if use_codec_proj:
        model_name = model_name.lower()
        if "qwen3" in model_name:
            return CodecQwen3ForCausalLM, CodecQwen3Config
        else:
            raise ValueError(f"Codec projection for {model_name} not supported.")
    else:
        return AutoModelForCausalLM, AutoConfig

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: str | None = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: str | None = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    dtype: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    codec_embed_file: str | None = field(
        default=None,
        metadata={
            "help": (
                "The codebook weights to use for codec projection with CodecQwen3ForCausalLM. If None, codec projection is not used."
            )
        },
    )
    unicode_offset: int = field(
        default=UNICODE_OFFSET_LARGE,
        metadata={
            "help": (
                "The unicode index of the token at the start of the codec vocabulary. "
                "This is used to determine which tokens should get codec projection in CodecQwen3ForCausalLM."
            )
        },
    )
    end_header_token: str = field(
        default="<|end_header|>",
        metadata={
            "help": "The token indicating the end of the header in the input sequences."
        },
    )
    pad_vocab_to_multiple_of: int = field(
        default=8,
        metadata={
            "help": (
                "Pad the vocabulary size to be a multiple of this value. Useful for optimizing tensor cores."
            )
        },
    )
    cache_dataset_only: bool = field(
        default=False,
        metadata={
            "help": (
                "If set to True, the dataset will be cached but the training / validation will not be run."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: str | None = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: str | None = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=False, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    pad_sequences_to_multiple_of: int = field(
        default=8,
        metadata={
            "help": (
                "Pad the sequences to be a multiple of this value. Useful for optimizing tensor cores."
            )
        },
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["txt"], "`train_file` should be a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["txt"], "`validation_file` should be a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can provide TXT training and evaluation files
    data_files = {}
    dataset_args = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    elif training_args.do_train:
        raise ValueError("Training file is required when training.")
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    elif training_args.do_eval:
        raise ValueError("Validation file is required when evaluating.")
    extension = (
        data_args.train_file.split(".")[-1]
        if data_args.train_file is not None
        else data_args.validation_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
    else:
        raise ValueError("Unsupported file extension. Only .txt files are supported.")
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        **dataset_args,
    )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    model_cls, config_cls = get_model_and_config_class(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        bool(model_args.codec_embed_file),
    )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    codec_embed_weight = None
    if model_args.codec_embed_file is not None:
        codec_embed_weight = torch.load(model_args.codec_embed_file, map_location="cpu")
        assert codec_embed_weight.ndim == 3, (
            "codec_embed_file must contain a tensor of shape (num_codebooks, codebook_size, codebook_dim)"
        )
        config_kwargs.update({
            "num_codebooks": codec_embed_weight.shape[0],
            "codebook_size": codec_embed_weight.shape[1],
            "codebook_dim": codec_embed_weight.shape[2],
            "tie_word_embeddings": False,
        })
        # Flatten along the codebook axis to match the model embedding weight matrix shape
        codec_embed_weight = codec_embed_weight.view(-1, codec_embed_weight.shape[-1])
    if model_args.config_name:
        config = config_cls.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = config_cls.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError(
            "You must specify either --config_name or --model_name_or_path."
        )

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        if not model_args.model_name_or_path:
            pad_to_multiple_of = model_args.pad_vocab_to_multiple_of
            # set the model vocabulary size to match the tokenizer, plus a multiple of `pad_to_multiple_of` as done in
            # https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/modeling_utils.py -> _get_resized_embeddings
            new_vocab_size = ((len(tokenizer) + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            pad_size = new_vocab_size - len(tokenizer)
            logger.info(f"Updating `config.vocab_size` ({config.vocab_size}) to `len(tokenizer) + {pad_size}` ({new_vocab_size}).")
            config.vocab_size = new_vocab_size
            config.bos_token_id = tokenizer.bos_token_id
            config.eos_token_id = tokenizer.eos_token_id
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if codec_embed_weight is not None:
        config.codec_vocab_start = tokenizer.convert_tokens_to_ids(chr(model_args.unicode_offset))
        logger.info(
            f"Setting `config.codec_vocab_start` to {config.codec_vocab_start} "
            f"({tokenizer.convert_ids_to_tokens(config.codec_vocab_start)})"
        )

    if not model_args.cache_dataset_only:
        if model_args.model_name_or_path:
            dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
            model = model_cls.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                dtype=dtype,
            )
        else:
            if model_cls is AutoModelForCausalLM:
                model = model_cls.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else:
                model = model_cls._from_config(config)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")

        # Sanity check on embedding size vs tokenizer vocab
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            raise ValueError(
                f"The tokenizer's vocabulary size ({len(tokenizer)}) is larger than the model's embedding size ({embedding_size}). "
                "Please ensure that the model's embedding layer is correctly sized to accommodate the tokenizer's vocabulary."
            )
        
        if codec_embed_weight is not None:
            model.set_codec_embeddings(codec_embed_weight.to(model.dtype))
            logger.info(
                f"Overwrote codec embeddings with {codec_embed_weight.dtype} tensor of shape {codec_embed_weight.shape}."
            )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            include = labels != -100
            return metric.compute(predictions=preds[include], references=labels[include])

    if model_args.cache_dataset_only:
        logger.info("Caching dataset only, exiting now.")
        return

    # Initialize our Trainer
    data_collator = DataCollatorWithLossMasking(
        tokenizer=tokenizer,
        end_header_token=model_args.end_header_token,
        pad_to_multiple_of=data_args.pad_sequences_to_multiple_of,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()