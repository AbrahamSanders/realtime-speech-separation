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

After inference is complete, you can run evaluation using the CLI:

```bash
# Example: evaluate all mixture + separation + reference triples in `output/eval/dataset_magicodec_131k_60s_dev_metadata`
rtss.eval \
    --input_dir=output/eval/dataset_magicodec_131k_60s_dev_metadata \
    --overwrite
```
Use --overwrite if you need to overwrite existing evaluation outputs, e.g. if testing new changes that will affect metric calculations for the same input files.
See `src/realtime_speech_separation/eval_cli.py` for all available command-line options and their descriptions.

For each separation output, the evaluation metrics will be saved in a corresponding JSON file, e.g. `1_results_dcaf4f46.json`, `2_results_dcaf4f46.json`, and so on.
The hash in the JSON filename matches the hash used in the corresponding separated output, ensuring that the evaluation results can be traced back to the exact inference arguments used.

Once complete, summary statistics for the results will be saved in `input_dir/summary_<hash>.json`, where the hash matches the one used in the separated output and evaluation result filenames. A CSV file containing all the file-level metrics will also be saved at `input_dir/metrics_<hash>.csv`.

The final file structure looks like:
```
input_dir/
├── dataset_a/
│   ├── 1.wav
│   ├── 1_enrollment.wav
│   ├── 1_reference.wav
|   ├── 1_separated_dcaf4f46.wav
|   ├── 1_results_dcaf4f46.json
│   ├── 2.wav
│   ├── 2_enrollment.wav
│   ├── 2_reference.wav
│   ├── 2_separated_dcaf4f46.wav
│   ├── 2_results_dcaf4f46.json
│   └── ...
├── dataset_b/
│   ├── 1.wav
│   ├── 1_enrollment.wav
│   ├── 1_reference.wav
│   ├── 1_separated_dcaf4f46.wav
│   ├── 1_results_dcaf4f46.json
│   ├── 2.wav
│   ├── 2_enrollment.wav
│   ├── 2_reference.wav
│   ├── 2_separated_dcaf4f46.wav
│   ├── 2_results_dcaf4f46.json
│   └── ...
├── ...
├── args_dcaf4f46.json
├── metrics_dcaf4f46.csv
└── summary_dcaf4f46.json
```

## Interpret the results

The summary JSON file (`summary_<hash>.json`) contains aggregated statistics for all evaluated files, such as average metrics across the dataset.
The metrics CSV file (`metrics_<hash>.csv`) contains file-level metrics for each separation output, allowing you to analyze performance on individual files.

You can use these files to understand the overall performance of your separation model and identify specific cases where it performs well or poorly.
The evaluation protocol with metric definitions can be found in [docs/evaluation-protocol.md](../../../docs/evaluation-protocol.md).