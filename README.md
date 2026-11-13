# Realtime Speech Separation (RTSS)

A real-time streaming Target Speaker Extraction (TSE) system based on an autoregressive codec LM.

The system performs an enrollment-conditioned speech separation task, converting mono-channel speech mixtures
into stereo audio with the target (T) speaker isolated on the left channel. All other residual (R) speakers
end up on the right channel:

```text
# Two-speaker Mixture:
Enroll(S1): Mono (S1+S2) -> Stereo (S1, S2)
Enroll(S2): Mono (S1+S2) -> Stereo (S2, S1)

# Three-speaker Mixture:
Enroll(S1): Mono (S1+S2+S3) -> Stereo (S1, S2+S3)
Enroll(S2): Mono (S1+S2+S3) -> Stereo (S2, S1+S3)
Enroll(S3): Mono (S1+S2+S3) -> Stereo (S3, S1+S2)

# N-speaker Mixture:
Enroll(S1): Mono (S1+...+SN) -> Stereo (S1, S2+...+SN)
Enroll(S2): Mono (S1+...+SN) -> Stereo (S2, S1+S3+...+SN)
...
Enroll(SN): Mono (S1+...+SN) -> Stereo (SN, S1+...+S(N-1))
```

## Installation

The package supports Python 3.11+. In your preferred Python environment, install
the package as follows, optionally including development dependencies:

```bash
# Typical install
python -m pip install -e .

# Development install
python -m pip install -e '.[dev]'
```

## Usage

### Inference CLI

```bash
# Example: separate all mixture + enrollment pairs in `output/eval/dataset_magicodec_131k_60s_dev_metadata`
# Use checkpoint Qwen3-1.7B-magicodec-no-bpe-131k-test/Qwen3-1.7B-magicodec-no-bpe-131k-test-F16.gguf, 
# a 0.8 second lookahead, and a fixed seed of 42:
rtss.separate \
    --input_dir=output/eval/dataset_magicodec_131k_60s_dev_metadata \
    --llm_model_path=Qwen3-1.7B-magicodec-no-bpe-131k-test/Qwen3-1.7B-magicodec-no-bpe-131k-test-F16.gguf \
    --delay_secs=0.8 \
    --seed=42
```

### Evaluation CLI

After inference is complete, you can run evaluation using the CLI:

```bash
# Example: evaluate all mixture + separation + reference triples in `output/eval/dataset_magicodec_131k_60s_dev_metadata`
rtss.eval \
    --input_dir=output/eval/dataset_magicodec_131k_60s_dev_metadata
```

### Usage Notes

By default, `rtss.separate` and `rtss.eval` will skip any already processed files.
To force re-processing, you can use the `--overwrite` flag with either command.

You can also use the `--input_filter` flag to specify a subset of input files to process. 
This can be useful when you only want to re-run inference or evaluation on specific files.
This filter matches any part of the full input file path. For example:
```bash
rtss.separate \
    --input_dir=output/eval/dataset_magicodec_131k_60s_dev_metadata \
    --input_filter "CallFriend_deu/5178" \
    --overwrite
```
will re-process only the inputs within `output/eval/dataset_magicodec_131k_60s_dev_metadata/CallFriend_deu/5178`

## Development

Run the checks in this environment:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

Direct runtime dependencies can be found in `pyproject.toml`. 
The `dev` extra installs pytest & Ruff. Tests run offline and generate their audio in
temporary directories. CI runs the same checks on Python 3.11 and 3.14.

[MIT license](LICENSE).
