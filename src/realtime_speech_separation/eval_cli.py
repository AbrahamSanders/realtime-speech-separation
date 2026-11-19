"""Discover evaluation audio and run the speech separation evaluation CLI."""

import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm

from .evaluation.audio import EVALUATION_SAMPLE_RATE
from .evaluation.pipeline import EvaluationUnavailableError, create_pipeline
from .evaluation.reporting import group_results, has_failures, write_reports
from .evaluation.types import ALL_METRICS
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


def run_eval(args: argparse.Namespace) -> bool:
    """Evaluate selected triples and reports; return whether any input/backend failed."""
    if not Path(args.input_dir).is_dir():
        raise EvaluationUnavailableError(f"Input directory does not exist or is not a directory: {args.input_dir}")
    pipeline = create_pipeline(Path(args.input_dir), args.metrics, speaker_model_path=args.speaker_model_path)
    metrics = pipeline.configuration.metrics
    logger.warning("Running a partial protocol evaluation with metrics: %s", ", ".join(metrics))
    input_files = get_input_files(args.input_dir, args.input_filter)
    mixture_stems = [Path(path).with_suffix("") for path in input_files]
    if len(mixture_stems) != len(set(mixture_stems)):
        raise EvaluationUnavailableError("Multiple mixture audio files share a stem in the same directory")
    results, cached_paths = [], set()
    for input_file in tqdm(input_files, desc="Processing input files"):
        separated_files = get_separated_files(input_file)
        separations = [(Path(path), get_inference_hash(input_file, path)) for path in separated_files]
        reference_file = get_reference_file(input_file)
        records, cached = pipeline.evaluate_mixture(
            Path(input_file), Path(reference_file), separations, overwrite=args.overwrite
        )
        results.extend(records)
        cached_paths.update(cached)
    if not results:
        logger.warning("No mixture/separation pairs matched the selected inputs")
    selected_mixtures = [os.path.relpath(path, args.input_dir) for path in input_files]
    for (inference_hash, _), records in group_results(results).items():
        paths = {record.sources.separated.path for record in records}
        write_reports(
            Path(args.input_dir),
            records,
            configuration=pipeline.configuration,
            inference_hash=inference_hash,
            filter_terms=args.input_filter or (),
            selected_mixtures=selected_mixtures,
            cached_result_paths=cached_paths & paths,
        )
    failed = any(has_failures(result) for result in results)
    if failed:
        logger.error("Evaluation contains invalid inputs or backend failures; see result JSONs and coverage")
    return failed


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
        help="Compatibility option; the protocol requires 16000 Hz.",
    )
    input_group.add_argument(
        "--compute_permutation",
        action="store_true",
        help="Unsupported: enrollment fixes the target/residual channel assignment.",
    )

    input_group.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        choices=["all", "core", *ALL_METRICS],
        help="Metric selection (default: all). Use core for the implemented partial protocol.",
    )
    input_group.add_argument(
        "--speaker_model_path",
        type=Path,
        help="Optional local WeSpeaker ONNX model path. By default, download and cache the VoxBlink2 model.",
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

    if args.compute_permutation:
        parser.error("The evaluation protocol does not allow --compute_permutation")

    log_level = logging.getLevelNamesMapping().get(args.log_level.upper())
    logging.basicConfig(level=log_level)
    logger.info(f"Running with args: {args}")

    try:
        return 1 if run_eval(args) else 0
    except (EvaluationUnavailableError, OSError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
