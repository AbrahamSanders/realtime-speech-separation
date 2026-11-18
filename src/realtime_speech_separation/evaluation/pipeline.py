"""Evaluate selected triples with shared CPU resources and mixture-local reuse."""

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .audio import AudioTriple, load_audio_triple
from .metrics import DnsmosP808, MetricScore, Pesq, SiSdr, Stoi, Visqol, centered_zero_energy
from .reporting import compatible_result, content_digest, evaluation_identity, result_path, write_result
from .types import (
    ALL_METRICS,
    IMPROVEMENT_METRICS,
    INTERVAL_METRICS,
    ActivityResult,
    AudioSource,
    AudioSources,
    Channel,
    ChannelResult,
    EvaluationConfig,
    Interval,
    IntervalMetricResult,
    IntervalSource,
    Metric,
    MetricResult,
    RecordingError,
    RecordingResult,
    SpeechRegion,
)
from .vad import (
    ACTIVITY_FRAME_SAMPLES,
    MERGE_GAP_SAMPLES,
    OVERLAP_MIN_ACTIVE_SAMPLES,
    OVERLAP_MIN_SPAN_SAMPLES,
    UTTERANCE_MIN_ACTIVE_SAMPLES,
    SileroVad,
    SpeechDetector,
    prepare_reference,
    prepare_separation,
)

logger = logging.getLogger(__name__)
CORE_METRICS: tuple[Metric, ...] = ("si_sdri", "overlap_si_sdri", "activity", "stoi", "stoii")
METRIC_FAMILIES = {
    "si_sdri": "si_sdr",
    "overlap_si_sdri": "si_sdr",
    "stoi": "stoi",
    "stoii": "stoi",
    "pesq": "pesq",
    "visqol": "visqol",
}
MetricCall = Callable[[NDArray[np.float32], NDArray[np.float32]], MetricScore]


class EvaluationUnavailableError(RuntimeError):
    """A selected metric family, package, or model is unavailable before processing."""


def _resolve_metrics(names: Sequence[str]) -> tuple[Metric, ...]:
    """Expand all/core and reject unavailable families without silently reducing scope."""
    selected = set()
    for name in names:
        if name == "all":
            selected.update(ALL_METRICS)
        elif name == "core":
            selected.update(CORE_METRICS)
        elif name in ALL_METRICS:
            selected.add(name)
        else:
            raise EvaluationUnavailableError(f"Unknown metric: {name}")
    unavailable = selected.difference(METRIC_FAMILIES, {"activity", "dnsmos_p808"})
    if unavailable:
        raise EvaluationUnavailableError(
            f"Requested metrics are not implemented: {', '.join(sorted(unavailable))}. "
            "Use --metrics core or individual implemented metrics for a partial protocol evaluation."
        )
    if not selected:
        raise EvaluationUnavailableError("Select at least one metric")
    return tuple(sorted(selected))


def create_pipeline(input_dir: Path, metrics: Sequence[str]) -> "EvaluationPipeline":
    """Preflight selected adapters and the packaged VAD once; never download assets."""
    metrics = _resolve_metrics(metrics)
    adapters = {}
    # Import and initialize only requested metric families, before any output writes.
    try:
        families = {METRIC_FAMILIES[name] for name in metrics if name in METRIC_FAMILIES}
        factories = {"si_sdr": SiSdr, "stoi": Stoi, "pesq": Pesq, "visqol": Visqol}
        for family in sorted(families):
            adapters[family] = factories[family]()
        dnsmos = DnsmosP808() if "dnsmos_p808" in metrics else None
        detector = SileroVad()
    except (ImportError, OSError, RuntimeError) as exc:
        raise EvaluationUnavailableError(f"Evaluation backend preflight failed: {exc}") from exc
    versions = {"numpy": version("numpy"), "librosa": version("librosa"), "soundfile": version("soundfile")}
    versions["torch"] = str(detector.torch_version)
    for adapter in adapters.values():
        versions.update(adapter.versions)
    model_digests = {"silero_vad.jit": detector.model_digest}
    if "visqol" in adapters:
        model_digests["visqol_speech.tflite"] = adapters["visqol"].model_digest
    metric_parameters = {name: adapter.parameters for name, adapter in adapters.items()}
    if dnsmos is not None:
        versions.update(dnsmos.versions)
        model_digests.update(dnsmos.model_digests)
        metric_parameters["dnsmos_p808"] = dnsmos.parameters
    # Persist the resolved threshold used for each selected interval adapter.
    metric_parameters = {
        name: {"min_active_samples": UTTERANCE_MIN_ACTIVE_SAMPLES, **parameters}
        for name, parameters in metric_parameters.items()
    }
    configuration = EvaluationConfig(
        metrics=metrics,
        parameters={
            "protocol_scope": "partial",
            "vad": detector.parameters,
            "timing": {
                "merge_gap_samples": MERGE_GAP_SAMPLES,
                "utterance_min_active_samples": UTTERANCE_MIN_ACTIVE_SAMPLES,
                "overlap_min_active_samples": OVERLAP_MIN_ACTIVE_SAMPLES,
                "overlap_min_span_samples": OVERLAP_MIN_SPAN_SAMPLES,
                "activity_frame_samples": ACTIVITY_FRAME_SAMPLES,
                "activity_majority": "strict_actual_frame_span",
            },
            **metric_parameters,
        },
        metric_backend_versions=versions,
        vad_backend_version=detector.backend_version,
        model_digests=model_digests,
    )
    return EvaluationPipeline(input_dir, configuration, detector, adapters, dnsmos)


def _interval_result(
    interval: Interval, source: IntervalSource, output: MetricScore, baseline: MetricScore | None = None
) -> IntervalMetricResult:
    """Retain available paired components even when an improvement is undefined."""
    if baseline is None:
        return IntervalMetricResult(interval, source, output.status, output.value, output.reason)
    input_score, output_score = baseline.value, output.value
    if output.status == baseline.status == "scored":
        difference = output_score - input_score
        return IntervalMetricResult(
            interval, source, "scored", difference, input_score=input_score, output_score=output_score
        )
    if output.status == "ineligible":
        status, reason = output.status, output.reason
    elif output.status == "failed" or baseline.status == "failed":
        status = "failed"
        reason = output.reason if output.status == "failed" else f"baseline_{baseline.reason}"
    elif output.status != "scored":
        status, reason = output.status, output.reason
    else:
        status, reason = "undefined", f"baseline_{baseline.reason}"
    return IntervalMetricResult(
        interval, source, status, reason=reason, input_score=input_score, output_score=output_score
    )


class EvaluationPipeline:
    """Sequential per-mixture evaluation with injectable VAD and interval adapters.

    Each call scopes reference timing and baseline caches to one mixture. Failed
    results are persisted for diagnosis and recomputed on a later invocation.
    """

    def __init__(
        self,
        input_dir: Path,
        configuration: EvaluationConfig,
        detector: SpeechDetector,
        adapters: dict[str, MetricCall],
        dnsmos: DnsmosP808 | None = None,
    ) -> None:
        """Keep shared resources and resolved identity; do not load audio yet."""
        self.input_dir = input_dir
        self.configuration = configuration
        self.evaluation_id = evaluation_identity(configuration)
        self.detector = detector
        self.adapters = adapters
        self.dnsmos = dnsmos

    def _source(self, path: Path) -> AudioSource:
        """Fingerprint bytes independently of decoding, retaining missing paths."""
        relative = os.path.relpath(path, self.input_dir)
        try:
            digest = content_digest(path)
        except OSError:
            digest = None
        return AudioSource(relative, digest)

    def evaluate_mixture(
        self, mixture: Path, reference: Path, separations: Sequence[tuple[Path, str]], *, overwrite: bool = False
    ) -> tuple[list[RecordingResult], set[str]]:
        """Load/reuse selected results, returning records and reused separation paths."""
        results, cached_paths = [], set()
        if not separations:
            return results, cached_paths
        inference_hashes = [inference_hash for _, inference_hash in separations]
        if len(inference_hashes) != len(set(inference_hashes)):
            raise EvaluationUnavailableError(f"Multiple separated audio files share an inference suffix for {mixture}")
        mixture_source, reference_source = self._source(mixture), self._source(reference)
        reference_speech = None
        baselines = {}
        for separated, inference_hash in separations:
            sources = AudioSources(mixture_source, reference_source, self._source(separated))
            destination = result_path(mixture, inference_hash, self.evaluation_id)
            if not overwrite:
                try:
                    payload = destination.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    payload = ""
                cached = compatible_result(
                    payload, configuration=self.configuration, inference_hash=inference_hash, sources=sources
                )
                if cached is not None:
                    results.append(cached)
                    cached_paths.add(cached.sources.separated.path)
                    continue

            # Validate the triple once before VAD or scoring and retain native metadata.
            audio = load_audio_triple(mixture, reference, separated)
            if isinstance(audio, RecordingError):
                error = replace(audio, path=os.path.relpath(audio.path, self.input_dir))
                logger.error("Invalid audio %s: %s", error.path, error.reason)
                result = RecordingResult(
                    inference_hash,
                    self.evaluation_id,
                    self.configuration,
                    sources,
                    None,
                    {"T": ChannelResult(), "R": ChannelResult()},
                    status="invalid",
                    error=error,
                )
            else:
                loaded_sources = tuple(
                    replace(
                        source, native_sample_rate=loaded.native_sample_rate, native_channels=loaded.native_channels
                    )
                    for source, loaded in zip(
                        (sources.mixture, sources.reference, sources.separated),
                        (audio.mixture, audio.reference, audio.separated),
                        strict=True,
                    )
                )
                sources = AudioSources(*loaded_sources)
                if reference_speech is None:
                    reference_speech = prepare_reference(audio.reference.samples, self.detector)
                channels = prepare_separation(audio.separated.samples, reference_speech, self.detector)
                vad_failed = False
                for channel, prepared in channels.items():
                    for role in ("reference", "separated"):
                        error = getattr(prepared, f"{role}_vad_error")
                        if error is not None:
                            vad_failed = True
                            logger.error("%s %s VAD failed for %s: %s", role, channel, separated, error)
                for index, channel in enumerate(("T", "R")):
                    channels[channel] = self._score_channel(
                        audio, channel, index, channels[channel], reference_speech.overlap, baselines
                    )
                result = RecordingResult(
                    inference_hash,
                    self.evaluation_id,
                    self.configuration,
                    sources,
                    audio.sample_count,
                    channels,
                    status="failed" if vad_failed else "processed",
                    reference_overlap=reference_speech.overlap,
                )
            write_result(destination, result)
            results.append(result)
        return results, cached_paths

    def _score_channel(
        self,
        audio: AudioTriple,
        channel: Channel,
        index: int,
        prepared: ChannelResult,
        reference_overlap: tuple[SpeechRegion, ...] | None,
        baselines: dict[tuple[str, Channel, Interval], MetricScore],
    ) -> ChannelResult:
        """Slice identical bounds without permutation and pool no recording means."""
        metrics = dict(prepared.metrics)
        outputs = {}
        for name in INTERVAL_METRICS:
            if name not in self.configuration.metrics:
                continue
            if prepared.status == "unavailable":
                metrics[name] = MetricResult("unavailable", reason="missing_output_channel")
                continue
            # Obtain the candidate regions and preserve failures of their VAD sources.
            if name == "dnsmos_p808":
                if prepared.separated_vad_error is not None:
                    metrics[name] = MetricResult("failed", reason="separated_vad_error")
                    continue
                regions, source = prepared.separated_utterances, "separated_utterance"
            elif name == "overlap_si_sdri":
                if reference_overlap is None:
                    metrics[name] = MetricResult("failed", reason="reference_overlap_vad_error")
                    continue
                regions, source = reference_overlap, "reference_overlap"
            else:
                if prepared.reference_vad_error is not None:
                    metrics[name] = MetricResult("failed", reason="reference_vad_error")
                    continue
                regions, source = prepared.reference_utterances, "reference_utterance"
            family = METRIC_FAMILIES.get(name, name)
            min_active_samples = self.configuration.parameters[family]["min_active_samples"]
            candidates = []
            for region in regions:
                interval = region.interval
                if name == "overlap_si_sdri" and interval.end - interval.start < OVERLAP_MIN_SPAN_SAMPLES:
                    candidates.append(
                        IntervalMetricResult(interval, source, "ineligible", reason="overlap_span_too_short")
                    )
                    continue
                if region.active_samples < min_active_samples:
                    candidates.append(
                        IntervalMetricResult(interval, source, "ineligible", reason="insufficient_activity")
                    )
                    continue
                if name == "dnsmos_p808":
                    assert self.dnsmos is not None
                    output_samples = audio.separated.samples[index, interval.start : interval.end]
                    score = self.dnsmos(output_samples)
                    candidates.append(_interval_result(interval, source, score))
                    continue
                key = (family, channel, interval)
                reference_samples = audio.reference.samples[index, interval.start : interval.end]
                if key not in outputs:
                    output_samples = audio.separated.samples[index, interval.start : interval.end]
                    outputs[key] = self.adapters[family](reference_samples, output_samples)
                baseline = None
                if name in IMPROVEMENT_METRICS:
                    # Share interval baselines across outputs without pooling T and R.
                    if key not in baselines:
                        mixture_samples = audio.mixture.samples[interval.start : interval.end]
                        baselines[key] = self.adapters[family](reference_samples, mixture_samples)
                    baseline = baselines[key]
                candidate = _interval_result(interval, source, outputs[key], baseline)
                candidates.append(candidate)
            metrics[name] = MetricResult("enabled", tuple(candidates))
        # Missing-utterance diagnostics do not depend on separated VAD detection.
        missing = ()
        if prepared.status == "available":
            missing = tuple(
                region.interval
                for region in prepared.reference_utterances
                if centered_zero_energy(audio.separated.samples[index, region.interval.start : region.interval.end])
            )
        activity = prepared.activity if "activity" in self.configuration.metrics else ActivityResult()
        return replace(prepared, metrics=metrics, activity=activity, separated_missing_utterances=missing)
