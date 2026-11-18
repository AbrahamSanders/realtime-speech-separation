"""Strict result persistence, identity checks, and pooled evaluation summaries.

Only the standard library and NumPy are needed. Discovery and selection happen
in the caller; summaries accept only the results reached through that selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import types
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Literal, Union, get_args, get_origin, get_type_hints

import numpy as np

from .types import (
    INTERVAL_METRICS,
    ActivityResult,
    AudioSources,
    ChannelResult,
    EvaluationConfig,
    IntervalMetricResult,
    MetricResult,
    RecordingResult,
    ReferenceEmbeddingExtractionResult,
    ReferencePrototypeExtractionResult,
    WordErrorResult,
)

PERCENTILE_METHOD = "linear"


def strict_json_dumps(value: object) -> str:
    """Serialize records or JSON values deterministically; reject NaN and infinity."""
    value = asdict(value) if is_dataclass(value) else value
    _validate_json(value)
    return json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def _validate_json(value: object) -> None:
    """Reject lossy key coercions and non-JSON values before encoding."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError("Expected finite JSON values and string object keys")


def _identity(value: object) -> str:
    """Hash canonical JSON without depending on dictionary insertion order."""
    return hashlib.sha256(strict_json_dumps(value).encode("utf-8")).hexdigest()


def evaluation_identity(configuration: EvaluationConfig) -> str:
    """Identify package, settings, and metric/VAD/speaker/ASR backends, excluding inputs."""
    return _identity({"configuration": asdict(configuration), "percentile_method": PERCENTILE_METHOD})


def normalized_filter_terms(terms: Sequence[str]) -> tuple[str, ...]:
    """Canonicalize the CLI's case-insensitive OR selection without changing terms."""
    return tuple(sorted({term.lower() for term in terms}))


def selection_identity(terms: Sequence[str] = ()) -> str:
    """Return 'all' for an unfiltered run, otherwise a digest of its filter terms."""
    normalized = normalized_filter_terms(terms)
    return _identity(normalized) if normalized else "all"


def content_digest(path: Path) -> str:
    """Stream a source/model SHA-256 digest; propagate file I/O errors to the caller."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous duplicate JSON keys, including inside configuration."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(value: object, expected: object) -> object:
    """Validate JSON against the dataclass contract, including nested field types.

    Require every field, even those with constructor defaults, so truncated or
    older records cannot silently acquire successful-looking default outcomes.
    """
    origin, arguments = get_origin(expected), get_args(expected)
    if expected is object:
        return value  # The complete payload has already passed JSON validation.
    if origin in (Union, types.UnionType):
        for alternative in arguments:
            try:
                return _decode(value, alternative)
            except (ValueError, TypeError):
                pass
        raise ValueError(f"Value does not match {expected}")
    if origin is Literal:
        if any(type(value) is type(option) and value == option for option in arguments):
            return value
    elif origin is tuple and isinstance(value, list):
        return tuple(_decode(item, arguments[0]) for item in value)
    elif origin is dict and isinstance(value, dict):
        return {_decode(key, arguments[0]): _decode(item, arguments[1]) for key, item in value.items()}
    elif is_dataclass(expected) and isinstance(value, dict):
        if set(value) != {field.name for field in fields(expected)}:
            raise ValueError(f"Missing or unknown fields for {expected.__name__}")
        hints = get_type_hints(expected)
        return expected(**{key: _decode(item, hints[key]) for key, item in value.items()})
    elif expected is float and type(value) in (int, float):
        return float(value)
    elif type(value) is expected:
        return value
    raise ValueError(f"Invalid value for {expected}")


def _validate_result(result: RecordingResult) -> None:
    """Check the identity of configuration, including the producing package version."""
    if result.evaluation_id != evaluation_identity(result.configuration):
        raise ValueError("Result evaluation ID does not match its configuration")


def parse_result(payload: str) -> RecordingResult:
    """Parse a complete compatible result; raise ValueError for malformed records."""
    value = json.loads(payload, object_pairs_hook=_object_pairs)
    _validate_json(value)
    result = _decode(value, RecordingResult)
    _validate_result(result)
    return result


def has_failures(result: RecordingResult) -> bool:
    """Detect recording or metric/VAD/speaker/ASR backend errors; allow ineligibility."""
    return result.status != "processed" or any(
        channel.reference_vad_error is not None
        or channel.separated_vad_error is not None
        or channel.activity.status == "failed"
        or channel.wer.status == "failed"
        or channel.prototype.status == "failed"
        or any(item.status == "failed" for item in channel.prototype.extractions)
        or any(item.status == "failed" for metric in channel.metrics.values() for item in metric.interval_results)
        for channel in result.channels.values()
    )


def _source_keys(sources: AudioSources) -> tuple[tuple[str, str | None], ...]:
    """Separate content/path validity from informational native audio metadata."""
    return tuple((source.path, source.sha256) for source in (sources.mixture, sources.reference, sources.separated))


def compatible_result(
    payload: str,
    *,
    configuration: EvaluationConfig,
    inference_hash: str,
    sources: AudioSources,
) -> RecordingResult | None:
    """Return reusable results, or None for stale/legacy/interrupted/failed JSON.

    File reading and overwrite policy belong to the caller. Missing source
    digests never establish reuse, even if two missing values happen to match.
    """
    try:
        result = parse_result(payload)
    except (ValueError, TypeError):
        return None
    if (
        result.evaluation_id != evaluation_identity(configuration)
        or result.inference_hash != inference_hash
        or _source_keys(result.sources) != _source_keys(sources)
        or any(digest is None for _, digest in _source_keys(sources))
        or has_failures(result)
    ):
        return None
    return result


def write_result(path: Path, result: RecordingResult) -> None:
    """Atomically replace a per-separation JSON; leave existing results on errors."""
    _validate_result(result)
    payload = strict_json_dumps(result)
    temporary = None
    try:
        # Keep the temporary file on the destination filesystem for atomic replace.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def result_path(mixture: Path, inference_hash: str, evaluation_id: str) -> Path:
    """Name adjacent results using the mixture stem and complete inference suffix."""
    return mixture.with_name(f"{mixture.stem}_results_{inference_hash}_{evaluation_id}.json")


def _recording_row(result: RecordingResult) -> dict[str, object]:
    """Flatten recording means/counts for CSV; corpus reports still pool intervals."""
    row = {
        "mixture": result.sources.mixture.path,
        "separated": result.sources.separated.path,
        "inference_hash": result.inference_hash,
        "evaluation_id": result.evaluation_id,
        "status": result.status,
        "error_path": result.error.path if result.error else None,
        "error_reason": result.error.reason if result.error else None,
    }
    for name in ("T", "R"):
        channel = result.channels[name]
        row[f"{name}_status"] = channel.status
        row[f"{name}_reference_vad_error"] = channel.reference_vad_error
        row[f"{name}_separated_vad_error"] = channel.separated_vad_error
        row[f"{name}_reference_active"] = channel.reference_active
        row[f"{name}_separated_missing_utterances"] = len(channel.separated_missing_utterances)
        summary = _channel_summary([channel], [result.sample_count or 0])
        for metric, measurements in summary["metrics"].items():
            prefix = f"{name}_{metric}"
            row[f"{prefix}_status"] = channel.metrics[metric].status
            row[f"{prefix}_mean"] = measurements["distribution"]["mean"]
            for key, value in measurements["coverage"].items():
                if key != "reasons":
                    row[f"{prefix}_{key}"] = value
            for component in ("input_scores", "output_scores"):
                if component in measurements:
                    row[f"{prefix}_{component}_mean"] = measurements[component]["mean"]
        for metric in ("activity", "wer"):
            row[f"{name}_{metric}_status"] = getattr(channel, metric).status
            row[f"{name}_{metric}_reason"] = getattr(channel, metric).reason
            for key, value in summary[metric].items():
                if not isinstance(value, dict):
                    row[f"{name}_{metric}_{key}"] = value
            for key, value in summary[metric]["coverage"].items():
                row[f"{name}_{metric}_{key}"] = value
    return row


def write_reports(
    input_dir: Path,
    results: Sequence[RecordingResult],
    *,
    configuration: EvaluationConfig,
    inference_hash: str,
    filter_terms: Sequence[str] = (),
    selected_mixtures: Sequence[str] = (),
    cached_result_paths: Collection[str] = (),
) -> None:
    """Write selection-specific strict JSON and one CSV row per reached separation."""
    summary = summarize_results(
        results,
        configuration=configuration,
        inference_hash=inference_hash,
        filter_terms=filter_terms,
        selected_mixtures=selected_mixtures,
        cached_result_paths=cached_result_paths,
    )
    suffix = f"{inference_hash}_{evaluation_identity(configuration)}_{selection_identity(filter_terms)}"
    payload = strict_json_dumps(summary)
    (input_dir / f"summary_{suffix}.json").write_text(payload, encoding="utf-8")
    rows = [_recording_row(result) for result in results]
    fieldnames = list(rows[0]) if rows else ["mixture", "separated", "inference_hash", "evaluation_id", "status"]
    with (input_dir / f"metrics_{suffix}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ratio(numerator: int, denominator: int) -> float | None:
    """Use null for undefined denominators rather than inventing a zero score."""
    return numerator / denominator if denominator else None


def coverage(results: Sequence[IntervalMetricResult | ReferenceEmbeddingExtractionResult]) -> dict[str, object]:
    """Derive candidate/eligible/scored counts and reasons from interval outcomes.

    Undefined and failed candidates have passed eligibility; only ineligible
    results leave that denominator. Prototype extractions use this same
    accounting but are reported separately from separated similarity scores.
    """
    counts = Counter(item.status for item in results)
    candidates = len(results)
    eligible = candidates - counts["ineligible"]
    return {
        "candidates": candidates,
        "eligible": eligible,
        **{status: counts[status] for status in ("scored", "ineligible", "undefined", "failed")},
        "scored_per_candidate": _ratio(counts["scored"], candidates),
        "scored_per_eligible": _ratio(counts["scored"], eligible),
        "reasons": {
            status: dict(sorted(Counter(item.reason for item in results if item.status == status).items()))
            for status in ("ineligible", "undefined", "failed")
        },
    }


def distribution(values: Sequence[float]) -> dict[str, object]:
    """Summarize pooled scores with linear percentiles and population SD (ddof=0)."""
    names = ("min", "p10", "q1", "median", "q3", "p90", "max")
    if not values:
        return {"count": 0, "mean": None, "sd": None, **dict.fromkeys(names)}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Distributions require finite scored interval results")
    percentiles = np.percentile(array, (0, 10, 25, 50, 75, 90, 100), method=PERCENTILE_METHOD)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=0)),
        **dict(zip(names, percentiles.tolist(), strict=True)),
    }


def _status_counts(
    items: Sequence[
        MetricResult | ActivityResult | WordErrorResult | ReferencePrototypeExtractionResult | ChannelResult
    ],
    statuses: Sequence[str],
) -> dict[str, int]:
    """Include zero-count statuses so empty summaries retain a stable shape."""
    counts = Counter(item.status for item in items)
    return {status: counts[status] for status in statuses}


def _count_result_coverage(items: Sequence[ActivityResult | WordErrorResult]) -> dict[str, object]:
    """Report activity/WER recording coverage without counting disabled channels."""
    candidates = sum(item.status in ("scored", "failed") for item in items)
    scored = sum(item.status == "scored" for item in items)
    return {
        "candidate_recordings": candidates,
        "scored_recordings": scored,
        "scored_per_candidate": _ratio(scored, candidates),
    }


def _channel_summary(channels: Sequence[ChannelResult], sample_counts: Sequence[int]) -> dict[str, object]:
    """Pool one channel's interval scores and raw counts, preserving partial successes."""
    metrics = {}
    for name in INTERVAL_METRICS:
        records = [channel.metrics[name] for channel in channels]
        interval_results = [item for record in records for item in record.interval_results]
        metrics[name] = {
            "recordings_by_status": _status_counts(records, ("enabled", "disabled", "unavailable")),
            "coverage": coverage(interval_results),
            "distribution": distribution([item.value for item in interval_results if item.status == "scored"]),
        }
        if name in ("si_sdri", "overlap_si_sdri", "stoii"):
            for component in ("input_score", "output_score"):
                metrics[name][component + "s"] = distribution(
                    [getattr(item, component) for item in interval_results if getattr(item, component) is not None]
                )

    # Aggregate confusion/edit counts before computing ratios, never recording ratios.
    activities = [channel.activity for channel in channels]
    activity_counts = [item.counts for item in activities if item.status == "scored"]
    tp, fp, fn, frames = (sum(getattr(item, key) for item in activity_counts) for key in ("tp", "fp", "fn", "frames"))
    candidate_frames = sum(
        (samples + 1599) // 1600
        for item, samples in zip(activities, sample_counts, strict=True)
        if item.status in ("scored", "failed")
    )
    activity = {
        "recordings_by_status": _status_counts(activities, ("scored", "disabled", "unavailable", "failed")),
        "coverage": {
            **_count_result_coverage(activities),
            "candidate_frames": candidate_frames,
            "scored_frames": frames,
            "scored_per_candidate_frame": _ratio(frames, candidate_frames),
        },
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "frames": frames,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }
    words = [channel.wer for channel in channels]
    word_counts = [item.counts for item in words if item.status == "scored"]
    edits = {
        key: sum(getattr(item, key) for item in word_counts)
        for key in ("substitutions", "deletions", "insertions", "reference_words")
    }
    wer = {
        "recordings_by_status": _status_counts(words, ("scored", "disabled", "unavailable", "failed")),
        "coverage": {
            **_count_result_coverage(words),
            "scored_reference_words": edits["reference_words"],
            "empty_reference_recordings": sum(item.reference_words == 0 for item in word_counts),
        },
        **edits,
        "wer": _ratio(edits["substitutions"] + edits["deletions"] + edits["insertions"], edits["reference_words"]),
    }
    prototypes = [channel.prototype for channel in channels]
    return {
        "recordings_by_status": _status_counts(channels, ("available", "unavailable")),
        "inactive_reference_channels": sum(channel.reference_active is False for channel in channels),
        "active_output_frames_on_inactive_reference": sum(
            channel.activity.counts.tp + channel.activity.counts.fp
            for channel in channels
            if channel.reference_active is False and channel.activity.status == "scored"
        ),
        "separated_missing_utterance_count": sum(len(channel.separated_missing_utterances) for channel in channels),
        "metrics": metrics,
        "activity": activity,
        "wer": wer,
        "prototype": {
            "recordings_by_status": _status_counts(prototypes, ("available", "unavailable", "disabled", "failed")),
            "coverage": coverage([item for prototype in prototypes for item in prototype.extractions]),
        },
    }


def group_results(results: Sequence[RecordingResult]) -> dict[tuple[str, str], list[RecordingResult]]:
    """Group selected separations by inference hash and evaluation identity."""
    groups = {}
    for result in results:
        groups.setdefault((result.inference_hash, result.evaluation_id), []).append(result)
    return groups


def summarize_results(
    results: Sequence[RecordingResult],
    *,
    configuration: EvaluationConfig,
    inference_hash: str,
    filter_terms: Sequence[str] = (),
    selected_mixtures: Sequence[str] = (),
    cached_result_paths: Collection[str] = (),
) -> dict[str, object]:
    """Summarize a single identity group, including an explicitly known empty group.

    processed counts valid evaluated records, including reused and partially
    failed records. failed counts records with any metric/VAD/speaker/ASR backend
    failure; invalid is separate. cached_result_paths identifies the separations
    whose result JSONs were reused from disk during this CLI invocation. This
    bookkeeping does not mutate the cached result files.
    """
    evaluation_id = evaluation_identity(configuration)
    for result in results:
        _validate_result(result)
        if result.inference_hash != inference_hash or result.evaluation_id != evaluation_id:
            raise ValueError("Cannot pool different inference/evaluation identities")
    paths = [result.sources.separated.path for result in results]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate separation in summary")
    if not set(cached_result_paths) <= set(paths):
        raise ValueError("Cached result paths must belong to the selected results")
    mixtures = sorted(set(selected_mixtures) | {result.sources.mixture.path for result in results})
    valid = [result for result in results if result.status != "invalid"]
    return {
        "inference_hash": inference_hash,
        "evaluation_id": evaluation_id,
        "configuration": asdict(configuration),
        "percentile_method": PERCENTILE_METHOD,
        "selection_id": selection_identity(filter_terms),
        "selection": {"filter_terms": normalized_filter_terms(filter_terms), "mixture_paths": mixtures},
        "recordings": {
            "total": len(results),
            "processed": len(valid),
            "invalid": len(results) - len(valid),
            "reused": len(set(cached_result_paths)),
            "target_only": sum(result.output_channels == ("T",) for result in valid),
            "failed": sum(has_failures(result) for result in valid),
        },
        "channels": {
            channel: _channel_summary(
                [result.channels[channel] for result in valid], [result.sample_count for result in valid]
            )
            for channel in ("T", "R")
        },
    }
