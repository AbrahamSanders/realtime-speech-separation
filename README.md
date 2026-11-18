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
# Core audio utilities
python -m pip install -e .

# Development install (used by CI)
python -m pip install -e '.[dev]'
```

Install the extras needed for your workflow:

```bash
python -m pip install -e '.[inference]'  # rtss.separate
python -m pip install -e '.[client]'     # rtss.client (includes inference)
python -m pip install -e '.[eval]'       # rtss.eval, including wideband PESQ
python -m pip install -e '.[training]'   # scripts/training
python -m pip install -e '.[data]'       # scripts/data_prep and scripts/eval_prep
```

Inference, training, and codec-based data preparation also require the local
`MagiCodec` clone described in [the architecture](docs/architecture.md).
The base and development installs do not load models or require the optional
runtime stacks. Python support for extras depends on their dependencies.

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
    --input_dir=output/eval/dataset_magicodec_131k_60s_dev_metadata \
    --metrics core
```

`core` runs a partial evaluation: utterance and overlap SI-SDRi, speech activity,
STOI, and STOI improvement. Select individual metrics with, for example,
`--metrics si_sdri activity stoi`. The default `--metrics all` requests the full
[protocol](docs/evaluation-protocol.md). Requested metrics never fall back silently.

Evaluation uses the supplied mixture, reference, and separated audio and packaged
Silero VAD on CPU. Enrollment and dataset metadata are unnecessary. Channels keep
their target/residual assignment; target-only output is supported. Audio is
resampled to 16 kHz and must then have equal lengths and already be aligned.

Per-separation results are written beside the mixture as
`<file_id>_results_<inference_hash>_<evaluation_id>.json`. Reports in `input_dir`
are named `summary_<inference_hash>_<evaluation_id>_<selection_id>.json` and
`metrics_<inference_hash>_<evaluation_id>_<selection_id>.csv`. The evaluation ID
identifies settings and runtime/model versions; the selection ID is `all` or a
filter digest. Summaries pool interval scores and activity counts; the CSV has
one row per separation. Dataset-filtered reports coexist with full-set reports.
Results are reused only when the source content and configuration match and no
VAD, metric, speaker, or ASR backend failed. Invalid inputs and failures in any
of these backends return a nonzero exit status. Undefined scores and ineligible
intervals retain explicit reasons and coverage. Each channel measurement also
reports recording participation and error exclusions with reason counts; partial
interval failures preserve the remaining scores.

#### PESQ setup

The `eval` extra includes the native `pesq` wrapper. Building PESQ from PyPI
requires a C compiler and Python development headers. Pip's
isolated build installs Cython and NumPy for the extension. For tested Python
and NumPy versions, see the
[Stage 5 implementation notes](docs/plans/eval_cli.md#stage-5--add-pesq).

Select PESQ with `--metrics pesq` or add it to the core metrics with
`--metrics core pesq`.

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
