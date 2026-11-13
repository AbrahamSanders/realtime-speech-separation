from typing import List, Dict
import numpy as np
import pandas as pd
import argparse
import logging
import librosa
import os
import json
from tqdm import tqdm
from asteroid.metrics import get_metrics

from .utils.data_utils import get_audio_files

logger = logging.getLogger(__name__)

def get_enrollment_file(input_file: str) -> str:
    file_name, ext = os.path.splitext(input_file)
    return f"{file_name}_enrollment{ext}"

def get_reference_file(input_file: str) -> str:
    file_name, ext = os.path.splitext(input_file)
    return f"{file_name}_reference{ext}"

def get_separated_files(input_file: str) -> List[str]:
    # For input file "1.wav", separated files will be like "1_separated_<inference_args_hash>.wav".
    # There may be multiple separated files for a single input file. Also, separated files may 
    # use a different extension than the input file.
    file_base, _ = os.path.splitext(input_file)
    file_base = os.path.basename(file_base)
    separated_files = []
    for f in os.listdir(os.path.dirname(input_file)):
        if f.startswith(f"{file_base}_separated_"):
            separated_files.append(os.path.join(os.path.dirname(input_file), f))
    return separated_files

def get_results_file(separated_file: str) -> str:
    file_name, _ = os.path.splitext(separated_file)
    return file_name.replace("_separated_", "_results_") + ".json"

def read_results_if_exist(separated_file: str) -> Dict:
    results_file = get_results_file(separated_file)
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return {}

def write_results(separated_file: str, results: Dict):
    results_file = get_results_file(separated_file)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

def run_eval(args: argparse.Namespace) -> None:
    input_files = get_audio_files(
        args.input_dir, 
        filter_include=args.input_filter, 
        filter_exclude=["_enrollment.", "_separated_", "_reference."],
    )

    # Perform the evaluation
    inference_args_hash_dict = {}
    for input_file in tqdm(input_files, desc="Processing input files"):
        separated_files = get_separated_files(input_file)
        for sep_file in separated_files:
            inference_args_hash = os.path.splitext(os.path.basename(sep_file))[0].split("_")[-1]

            # Check if results already exist for this separation and load them if available. Otherwise proceed to compute the metrics
            results = read_results_if_exist(sep_file)
            if results and not args.overwrite:
                logger.info(f"Found results for separation {inference_args_hash} of input audio {input_file}. Skipping.")
            else:
                logger.info(f"Processing separation {inference_args_hash} of input audio {input_file}.")

                # Load the input, enrollment, reference, and separated audio files
                input_audio, _ = librosa.load(input_file, sr=args.sample_rate, mono=True, dtype=np.float32)
                enrollment_audio, _ = librosa.load(get_enrollment_file(input_file), sr=args.sample_rate, mono=True, dtype=np.float32)
                reference_audio, _ = librosa.load(get_reference_file(input_file), sr=args.sample_rate, mono=False, dtype=np.float32)
                separated_audio, _ = librosa.load(sep_file, sr=args.sample_rate, mono=False, dtype=np.float32)

                # Compute and write the evaluation metrics for the separated audio
                results = get_metrics(
                    mix=input_audio,
                    clean=reference_audio,
                    estimate=separated_audio,
                    sample_rate=args.sample_rate,
                    metrics_list=["si_sdr", "stoi"],
                    average=False,
                    compute_permutation=args.compute_permutation,
                    ignore_metrics_errors=True,
                    filename=sep_file,
                )
                for key in results:
                    if results[key] is not None:
                        results[key] = results[key].ravel()
                for key in list(results):
                    if not key.startswith("input_") and f"input_{key}" in results:
                        if results[key] is not None and results[f"input_{key}"] is not None:
                            results[f"{key}_delta"] = results[key] - results[f"input_{key}"]
                        else:
                            results[f"{key}_delta"] = None
                for key in results:
                    if results[key] is not None:
                        results[key] = results[key].tolist()
                write_results(sep_file, results)

            # Cache the results for this separation using the inference args hash
            if inference_args_hash not in inference_args_hash_dict:
                inference_args_hash_dict[inference_args_hash] = []
            inference_args_hash_dict[inference_args_hash].append((sep_file, results))

    # Log the aggregated results
    for inference_args_hash, results_list in inference_args_hash_dict.items():
        logger.info(f"Summarizing results for inference args {inference_args_hash}")
        results_summary = {}

        # Filter out results that contain None values to avoid errors during aggregation
        filtered_results_list = [(sep_file, results) for sep_file, results in results_list if not any(value is None for value in results.values())]

        # Compute the mean, sd and 5-number summary of each metric across all filtered results
        for key in filtered_results_list[0][1]:
            metric_vals = np.array([results[key] for _, results in filtered_results_list])
            results_summary[key] = {
                "mean": metric_vals.mean(axis=0).tolist(),
                "sd": metric_vals.std(axis=0).tolist(),
                "min": metric_vals.min(axis=0).tolist(),
                "max": metric_vals.max(axis=0).tolist(),
                "p10": np.percentile(metric_vals, 10, axis=0).tolist(),
                "q1": np.percentile(metric_vals, 25, axis=0).tolist(),
                "median": np.percentile(metric_vals, 50, axis=0).tolist(),
                "q3": np.percentile(metric_vals, 75, axis=0).tolist(),
                "p90": np.percentile(metric_vals, 90, axis=0).tolist(),
            }

        # Other stats
        results_summary["total_count"] = len(results_list)
        results_summary["excluded_count"] = len(results_list) - len(filtered_results_list)
        results_summary["compute_permutation"] = args.compute_permutation

        # Write summary to a JSON file
        summary_filename = os.path.join(args.input_dir, f"summary_{inference_args_hash}.json")
        with open(summary_filename, "w") as f:
            json.dump(results_summary, f, indent=4)

        # Write a CSV with the metrics for each separated file in filtered_results_list
        csv_filename = os.path.join(args.input_dir, f"metrics_{inference_args_hash}.csv")
        csv_rows = []
        for sep_file, results in filtered_results_list:
            row = {"sep_file": os.path.relpath(sep_file, args.input_dir)}
            for key in results:
                for i, val in enumerate(results[key]):
                    row[f"{key}_{'T' if i == 0 else 'R'}"] = val
            csv_rows.append(row)
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_filename, index=False)

def main(argv: list[str] | None = None) -> int:
    """Run the Speech Separation Evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)

    input_group = parser.add_argument_group("Input Arguments")
    input_group.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing input/enrollment/reference/separated pairs.")
    input_group.add_argument("--input_filter", type=str, nargs="+", help="Filter for input files.")
    input_group.add_argument("--sample_rate", type=int, default=16000, help="Sample rate to use when loading audio files.")
    input_group.add_argument("--compute_permutation", action="store_true", help="Whether to compute the optimal permutation for separated sources.")

    output_group = parser.add_argument_group("Output Arguments")
    output_group.add_argument("--overwrite", action="store_true", help="Overwrite existing results files (summaries are always recomputed).")

    logging_group = parser.add_argument_group("Logging Arguments")
    logging_group.add_argument("--log_level", type=str, default="WARNING", choices=list(logging.getLevelNamesMapping()), help="Logging level.")
    
    args = parser.parse_args(argv)

    log_level = logging.getLevelNamesMapping().get(args.log_level.upper())
    logging.basicConfig(level=log_level)
    logger.info(f"Running with args: {args}")

    run_eval(args)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
