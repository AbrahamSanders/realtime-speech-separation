"""Discover evaluation audio and run the speech separation evaluation CLI."""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .evaluation.audio import EVALUATION_SAMPLE_RATE, load_audio_triple
from .evaluation.types import RecordingError
from .utils.data_utils import RAW_AUDIO_EXTENSIONS, get_audio_files

logger = logging.getLogger(__name__)


def get_input_files(input_dir: str, input_filter: str | list[str] | None = None) -> list[str]:
    """Find mixtures recursively with case-insensitive full-path substring filters.

    Multiple terms use OR semantics. Companion markers are excluded from paths;
    metric, VAD, speaker, and ASR backend assets belong outside the input tree.
    """
    return get_audio_files(
        input_dir,
        filter_include=input_filter,
        filter_exclude=["_enrollment.", "_separated_", "_reference."],
    )


def get_reference_file(input_file: str) -> str:
    """Return the adjacent reference path with the mixture's extension."""
    file_name, ext = os.path.splitext(input_file)
    return f"{file_name}_reference{ext}"


def get_separated_files(input_file: str) -> list[str]:
    """Find adjacent audio outputs, ignoring sidecars, directories, and empty IDs."""
    mixture = Path(input_file)
    prefix = f"{mixture.stem}_separated_"
    return sorted(
        str(path)
        for path in mixture.parent.iterdir()
        if path.is_file()
        and path.suffix.lower() in RAW_AUDIO_EXTENSIONS
        and path.stem.startswith(prefix)
        and len(path.stem) > len(prefix)
    )


def get_inference_hash(input_file: str, separated_file: str) -> str:
    """Parse the entire inference suffix relative to the mixture stem."""
    prefix = f"{Path(input_file).stem}_separated_"
    stem = Path(separated_file).stem
    if not stem.startswith(prefix) or len(stem) == len(prefix):
        raise ValueError(f"Separation {separated_file} does not match mixture {input_file}")
    return stem[len(prefix) :]


def get_results_file(separated_file: str) -> str:
    file_name, _ = os.path.splitext(separated_file)
    return file_name.replace("_separated_", "_results_") + ".json"


def read_results_if_exist(separated_file: str) -> dict:
    results_file = get_results_file(separated_file)
    if os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)
    return {}


def write_results(separated_file: str, results: dict):
    results_file = get_results_file(separated_file)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)


def run_eval(args: argparse.Namespace) -> bool:
    """Evaluate selected triples; return whether an audio input was invalid."""
    import pandas as pd
    from asteroid.metrics import get_metrics

    input_files = get_input_files(args.input_dir, args.input_filter)
    invalid_input = False

    # Perform the evaluation
    inference_args_hash_dict = {}
    for input_file in tqdm(input_files, desc="Processing input files"):
        separated_files = get_separated_files(input_file)
        for sep_file in separated_files:
            inference_args_hash = get_inference_hash(input_file, sep_file)

            # Check if results already exist for this separation and load them if available. Otherwise proceed to compute the metrics
            results = read_results_if_exist(sep_file)
            if results and not args.overwrite:
                logger.info(
                    f"Found results for separation {inference_args_hash} of input audio {input_file}. Skipping."
                )
            else:
                logger.info(f"Processing separation {inference_args_hash} of input audio {input_file}.")

                # Validate the supplied triple without enrollment or delay correction.
                audio = load_audio_triple(input_file, get_reference_file(input_file), sep_file)
                if isinstance(audio, RecordingError):
                    logger.error("Invalid audio %s: %s", audio.path, audio.reason)
                    invalid_input = True
                    continue
                separated_audio = audio.separated.samples

                # Compute and write the evaluation metrics for the separated audio
                results = get_metrics(
                    mix=audio.mixture.samples,
                    clean=audio.reference.samples[: separated_audio.shape[0]],
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
        filtered_results_list = [
            (sep_file, results)
            for sep_file, results in results_list
            if not any(value is None for value in results.values())
        ]

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

    return invalid_input


def main(argv: list[str] | None = None) -> int:
    """Run the Speech Separation Evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)

    input_group = parser.add_argument_group("Input Arguments")
    input_group.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing mixture/reference/separated triples.",
    )
    input_group.add_argument("--input_filter", type=str, nargs="+", help="Filter for input files.")
    input_group.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate to use when loading audio files.",
    )
    input_group.add_argument(
        "--compute_permutation",
        action="store_true",
        help="Whether to compute the optimal permutation for separated sources.",
    )

    output_group = parser.add_argument_group("Output Arguments")
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results files (summaries are always recomputed).",
    )

    logging_group = parser.add_argument_group("Logging Arguments")
    logging_group.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        choices=list(logging.getLevelNamesMapping()),
        help="Logging level.",
    )

    args = parser.parse_args(argv)
    if args.sample_rate != EVALUATION_SAMPLE_RATE:
        parser.error("The evaluation protocol requires --sample_rate=16000")

    log_level = logging.getLevelNamesMapping().get(args.log_level.upper())
    logging.basicConfig(level=log_level)
    logger.info(f"Running with args: {args}")

    return 1 if run_eval(args) else 0


if __name__ == "__main__":
    raise SystemExit(main())
