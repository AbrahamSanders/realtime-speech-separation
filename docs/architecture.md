# Architecture

## Model Architecture

We train an unmodified [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) backbone to perform the target speaker extraction task.
Audio is represented as discrete tokens from [MagiCodec](https://github.com/Ereboas/MagiCodec). Given an 2-10 second enrollment sample, the model
learns to predict target (T) and residual (R) tokens for every mixture (M) token.

There is a controllable delay between the input mixture tokens and the predicted target and residual tokens to allow the model to leverage future context for more accurate separation. This is typically referred to as the model's "lookahead" and is a key factor in balancing latency and separation quality.

During training, MagiCodec's quantizer embeddings are projected into Qwen3's embedding space with a lightweight linear projection layer.
This allows us to avoid re-learning the geometry of MagiCodec's encoder embedding space, which already captures the structure of the audio representation effectively.
After training we persist the projection of each token into the Qwen3 embedding matrix, allowing the model to be inferenced as a vanilla LM with no architectural modifications.

The model is developed using [HuggingFace Transformers](https://github.com/huggingface/transformers), and served with [Llama.cpp](https://github.com/abetlen/llama-cpp-python).
The Qwen3 wrapper containing the codec embedding projections used during training can be found in [codec_qwen3.py](../src/realtime_speech_separation/codec_qwen3.py).

### Input:

The sequence format is:
```text
<|delay_XXXms|><|target_voice|>...mono-channel input enrollment tokens, 2-10 seconds...<|end_header|>MTRMTRMTRMTR... 60 seconds of mixture-target-residual triples.
```
At each frame, the mixture token is provided from input and the model generates the next two tokens, corresponding to the target and residual audio.

Each MagiCodec frame corresponds to 0.02 seconds (20 ms) of audio, with a framerate of 50 frames per second.

The following delay tokens are supported:
- `<|delay_0ms|>`: No lookahead, the model only uses past context.
- `<|delay_100ms|>`: 5-frame lookahead.
- `<|delay_200ms|>`: 10-frame lookahead.
- `<|delay_400ms|>`: 20-frame lookahead.
- `<|delay_800ms|>`: 40-frame lookahead.
- `<|delay_1600ms|>`: 80-frame lookahead.
- `<|delay_3200ms|>`: 160-frame lookahead.
- `<|delay_6400ms|>`: 320-frame lookahead.

When a non-zero delay is provided representing a K-frame lookahead, the first K frames will have placeholder tokens for the target and residual audio, as the model cannot generate them until it has seen K future frames:

```text
<|delay_100ms|><|target_voice|>...mono-channel input enrollment tokens, 2-10 seconds...<|end_header|>M--M--M--M--M--MTRMTRMTRMTR...
```

Here we have a 5-frame lookahead, so the first generated T and R tokens in frame 6 correspond with the first M token in frame 1 (immediately after <|end_header|>).

For tokenization and data storage convenience, we use [codec-bpe](https://github.com/AbrahamSanders/codec-bpe) to create a tokenizer that represents each audio
token index as a distinct unicode character (No BPE learning is actually done, it is purely a one-to-one mapping). An real example sequence looks like this:

```text
<|delay_100ms|><|target_voice|>𣻹𜙡𘈔𫷄𔬻𔬻𓘿𐂟𠨅...𭖟𢯠𞳳𔐌𖂐𨺑𔰺𧖿🅝<|end_header|>𤀫--𞋗--𠒃--𔥣--𜘀--𛦂𣣗𡷸😆𜓿𞵰𞋿𘔫𠒏𙋊🖉😆𠴿𫆣𕱹𜣈𪝛𦇌𤸧𤏾...
```

## Training Data

Training data is sourced from the following datasets:

### Two-speaker conversations

Two-party phone calls or video calls recorded with one speaker per channel.

- [CallFriend](https://talkbank.org/ca/access/CallFriend)
- [CallHome](https://talkbank.org/ca/access/CallHome)
- [CANDOR](https://talkbank.org/ca/access/CANDOR.html)
- [Fisher English](https://catalog.ldc.upenn.edu/LDC2004S13)
- [Switchboard 1](https://catalog.ldc.upenn.edu/LDC97S62)

### Multiparty conversations

Multiparty conversations (meetings, dinner parties) recorded with one speaker per channel from lapel microphones.
Preprocessing was done to suppress most microphone bleed in these isolated channels.

- [AMI meeting corpus](https://groups.inf.ed.ac.uk/ami/corpus/)

### Synthetic mixtures

Synthetic overlapping audiobook recordings created by mixing single-speaker audiobook segments.

- [LibriCSS](https://github.com/chenzhuo1011/libri_css)

## Training Pipeline

1. All source audio is prepared as 2-channel audio with the target speaker on the left and the residual speaker(s) on the right.
   The two-speaker data is already in this format natively; for the multiparty and synthetic mixtures, a separate stereo mix
   is created for each available speaker, with that speaker isolated on the left channel and all other speakers mixed on the right channel.
   Source audio is stored in `data/audio/raw`.
2. The source audio is encoded with MagiCodec, and the resulting codes are stored in numpy arrays, one per file per channel. These are stored in
   `data/audio/codes/MagiCodec-50Hz-Base/0.1s_2.0s/stereo`.
3. The source audio is loaded as mono (mixing the left and right channels), and the result is also encoded with MagiCodec. These are stored in
   `data/audio/codes/MagiCodec-50Hz-Base/0.1s_2.0s/mono`.
4. The [prep_lm_dataset_magicodec.sh](../prep_lm_dataset_magicodec.sh) script is used to construct a plain-text dataset from the encoded audio codes.
   The dataset is one-line-per example, covering all source audio with configurable overlap between consecutive examples and randomized enrollment selection.
   This is stored in `output/dataset_name.txt` with corresponding per-example metadata stored in `output/dataset_name_metadata.jsonl`. The examples are also
   split into train/dev/test subsets, stored as `output/dataset_name_train.txt`, `output/dataset_name_dev.txt`, and `output/dataset_name_test.txt`
   with corresponding metadata files. For example:
   ```bash
    ls -la output/

    -rw-r--r--  1 user group    19674112 Sep 18 16:00 dataset_magicodec_131k_60s_dev_metadata.jsonl
    -rw-r--r--  1 user group  3234055955 Sep 18 16:00 dataset_magicodec_131k_60s_dev.txt
    -rw-r--r--  1 user group     1513819 Sep 18 15:57 dataset_magicodec_131k_60s_file_splits.csv
    -rw-r--r--  1 user group   400957667 Sep 18 15:57 dataset_magicodec_131k_60s_metadata.jsonl
    -rw-r--r--  1 user group    40089050 Sep 18 16:00 dataset_magicodec_131k_60s_test_metadata.jsonl
    -rw-r--r--  1 user group  6588359813 Sep 18 16:00 dataset_magicodec_131k_60s_test.txt
    -rw-r--r--  1 user group   341194505 Sep 18 16:00 dataset_magicodec_131k_60s_train_metadata.jsonl
    -rw-r--r--  1 user group 56050083504 Sep 18 16:00 dataset_magicodec_131k_60s_train.txt
    -rw-r--r--  1 user group 65872499272 Sep 18 15:57 dataset_magicodec_131k_60s.txt
   ```
5. The codec-bpe tokenizer is created using [make_magicodec_tokenizer_no_bpe.sh](../make_magicodec_tokenizer_no_bpe.sh), and is saved in
   `output/magicodec_no_bpe_1cb_131k`.
6. Training is performed using [train.py](../scripts/training/train.py), a modified version of the standard HuggingFace `run_clm.py` script.
   This is typically done as a distributed job on an external compute cluster or cloud service. The trained model checkpoint should be placed 
   in the directory `model_name` (e.g. Qwen3-1.7B-magicodec-no-bpe-131k)
7. The model checkpoint is prepared for inference using [prep_test_model.sh](../prep_test_model.sh), pointed to the checkpoint directory `model_name`.
   This prep script persists the codec embedding projections into Qwen3's embedding layer and creates the GGML quantized model files 
   ready for inference with llama.cpp. It is output to a parallel folder `model_name-test`.

## Inference

Inference is performed using the Inference CLI `rtss.separate` (points to [inference_cli.py](../src/realtime_speech_separation/inference_cli.py)).
Usage is shown in [README.md](../README.md).

Every separated output audio file is tagged with a hash of the inference arguments used to generate it, allowing easy tieback of evaluation results
to the specific inference configuration used. This ensures reproducibility and traceability of evaluation results.

### Inference Components

- [audio_tokenizer.py](../src/realtime_speech_separation/audio_tokenizer.py) is a wrapper around MagiCodec and codec-bpe, allowing a raw waveform
  to be converted to unicode audio strings and decoded back to raw waveforms. AudioTokenizer contains a context buffer for maintaining high-fidelity
  streaming encoding/decoding.
- [realtime_separator.py](../src/realtime_speech_separation/realtime_separator.py) implements the real-time speech separation logic.
  Accepts fixed-size chunks of mono audio (typically 100ms) and outputs corresponding fixed size chunks of separated stereo audio.
  Internally, audio is processed frame-by-frame at a 20ms resolution. The input/output chunk size can be configured as needed for different
  latency and performance requirements. MagiCodec's decoding quality suffers once the chunk size is reduced below 100ms.
- [realtime_separator_config.py](../src/realtime_speech_separation/realtime_separator_config.py) contains the configuration settings for the real-time separator.
- [realtime_separator_resources.py](../src/realtime_speech_separation/realtime_separator_resources.py) contains the resource management logic 
  for the real-time separator, such as loading and caching codec & LM model weights.
- [realtime_separator_profiler.py](../src/realtime_speech_separation/realtime_separator_profiler.py) contains profiling and performance measurement logic 
  for the real-time separator, helping to identify bottlenecks and optimize processing efficiency.

## Evaluation Pipeline

Evaluation is performed using the Evaluation CLI `rtss.eval` (points to [eval_cli.py](../src/realtime_speech_separation/eval_cli.py)).
Usage is shown in [README.md](../README.md), and the evaluation protocol is described in [evaluation_protocol.md](evaluation_protocol.md).

[eval_inference_0.03.sh](../eval_inference_0.03.sh) chains the extraction of evaluation data from the source audio for 3% of the dev set, 
running inference, and computing the evaluation metrics with the evaluation CLI. This script requires a GPU to run, since it performs inference before evaluation.

In-depth description of the evaluation pipeline output can be found in the [run-eval skill](../.agents/skills/run-eval/SKILL.md).
