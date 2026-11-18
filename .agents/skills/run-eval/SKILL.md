---
name: run-eval
description: Run inference and evaluation using a specified mono mixture, enrollment, and stereo reference.
---

# Run inference and evaluation

Use `AGENTS.md` for the project environment.

## Run inference with the CLI

```bash
# Example: separate all mixture + enrollment pairs in `output/eval/dataset_magicodec_131k_60s_dev_metadata`
# Use checkpoint Qwen3-1.7B-magicodec-no-bpe-131k-test/Qwen3-1.7B-magicodec-no-bpe-131k-test-F16.gguf, 
# a 0.8 second lookahead, and a fixed seed of 42:
rtss.separate \
    --input_dir=output/eval/dataset_magicodec_131k_60s_dev_metadata \
    --llm_model_path=Qwen3-1.7B-magicodec-no-bpe-131k-test/Qwen3-1.7B-magicodec-no-bpe-131k-test-F16.gguf \
    --delay_secs=0.8 \
    --seed=42 \
    --overwrite
```
Use --overwrite if you need to overwrite existing separated outputs, e.g. if testing new changes that will affect separation results for the same input files and arguments.
See `src/realtime_speech_separation/inference_cli.py` for all available command-line options and their descriptions.

This will recurse through all subdirectories of --input_dir and process all mixture + enrollment pairs found within.
Example input file structure:
```
input_dir/
├── dataset_a/
│   ├── 1.wav
│   ├── 1_enrollment.wav
│   ├── 1_reference.wav
│   ├── 2.wav
│   ├── 2_enrollment.wav
│   ├── 2_reference.wav
│   └── ...
├── dataset_b/
│   ├── 1.wav
│   ├── 1_enrollment.wav
│   ├── 1_reference.wav
│   ├── 2.wav
│   ├── 2_enrollment.wav
│   ├── 2_reference.wav
│   └── ...
└── ...
```
For each mixture file, the separated output will be generated in-place. For example: [`1.wav`, `1_enrollment.wav`] → `1_separated_dcaf4f46.wav`.
The hash uniquely identifies the inference arguments used for separation so results can be reproduced consistently.
A JSON file containing the inference arguments will be saved at `input_dir/args_<hash>.json`, where the hash matches the one used in the separated output filenames.

The reference files are not used by `rtss.separate` during inference; they are only for evaluation.

## Run evaluation with the CLI

Evaluation needs aligned mixture/reference/separated triples; it does not load
enrollment audio, dataset metadata, dataset VAD, or separator resources.

```bash
rtss.eval \
    --input_dir=output/eval/dataset_magicodec_131k_60s_dev_metadata \
    --metrics core \
    --overwrite
```

`--metrics all` (the default) requests the full protocol and rejects unavailable
metrics. Use `core` for utterance and overlap SI-SDRi, activity, STOI, and STOIi,
or select individual metrics, for example `--metrics si_sdri activity stoi`.
Subsets are marked as partial protocol evaluations. See the
[README](../../../README.md#evaluation-cli) for current availability and CLI usage.

Use `--input_filter` to constrain processing and reports, including reused results.
For development smoke tests, first copy triples to a temporary directory, then
use `--input_filter` and `--overwrite` there. `--overwrite` recomputes results;
otherwise reuse requires matching source hashes and evaluation configuration,
structural compatibility, and no VAD, metric, speaker, or ASR backend failure.
Undefined scores and ineligible intervals remain reusable. Summaries are rebuilt
for the current selection. Nonzero exit status indicates invalid inputs,
configuration errors, file I/O errors, or requested backend failures.

## Interpret the results

For inference hash `dcaf4f46`, evaluation identity `<evaluation_id>`, and an
unfiltered selection, the output includes:

```text
input_dir/
├── dataset_a/
│   ├── 1.wav
│   ├── 1_reference.wav
│   ├── 1_separated_dcaf4f46.wav
│   └── 1_results_dcaf4f46_<evaluation_id>.json
├── dataset_b/
│   ├── 2.wav
│   ├── 2_reference.wav
│   ├── 2_separated_dcaf4f46.wav
│   └── 2_results_dcaf4f46_<evaluation_id>.json
├── args_dcaf4f46.json
├── metrics_dcaf4f46_<evaluation_id>_all.csv
└── summary_dcaf4f46_<evaluation_id>_all.json
```

Inference writes `args_<inference_hash>.json`. Evaluation results retain that hash
and add an evaluation ID covering package version, selected metrics, settings,
VAD/metric/speaker/ASR backend versions, and model digests. Source audio hashes
are separate reuse keys. Filtered reports replace `all` with a selection digest
and coexist with full-set reports; the summary records filter terms and selected
mixture paths.

Per-separation JSON contains source/configuration provenance, T/R availability,
raw VAD intervals and utterances, separate reference/separated VAD errors,
interval scores with statuses/reasons and paired improvement components,
activity/edit counts, and missing-output diagnostics. Mono output is target-only;
residual measurements are explicitly unavailable. Disabled metrics are explicit.

Summary JSON pools interval scores and sums raw activity/edit counts before
computing ratios. It includes distributions, metric-specific coverage and reasons,
and processed/invalid/reused/failed recording counts. The CSV has one row per
separation with channel statuses, scalar summaries, raw counts, and coverage;
its recording means must not be averaged to reconstruct corpus scores.

Metric definitions and aggregation rules live in the
[evaluation protocol](../../../docs/evaluation-protocol.md).
