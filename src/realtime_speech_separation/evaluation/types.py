"""Shared evaluation contracts; sample bounds are half-open at the evaluation rate.

These records contain no arrays, models, or metric/VAD/speaker/ASR backend imports. A result
contains both T and R channel records, including an unavailable R for mono output.
The package version identifies published result formats and evaluation behavior.
Developers changing records within a package version must regenerate local results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .. import __version__

Channel: TypeAlias = Literal["T", "R"]
Metric: TypeAlias = Literal[
    "si_sdri", "overlap_si_sdri", "stoi", "stoii", "pesq", "visqol", "dnsmos_p808", "speaker_cosine", "activity", "wer"
]
# Per-interval outcomes: a score/extraction succeeded, eligibility excluded it,
# the calculation is undefined, or the metric/speaker backend failed to execute.
IntervalResultStatus: TypeAlias = Literal["scored", "ineligible", "undefined", "failed"]
# Activity/WER outcomes apply to a full channel and its frame/word counts.
CountResultStatus: TypeAlias = Literal["scored", "disabled", "unavailable", "failed"]
IntervalSource: TypeAlias = Literal["reference_utterance", "reference_overlap", "separated_utterance"]
INTERVAL_METRICS: tuple[Metric, ...] = (
    "si_sdri",
    "overlap_si_sdri",
    "stoi",
    "stoii",
    "pesq",
    "visqol",
    "dnsmos_p808",
    "speaker_cosine",
)
ALL_METRICS: tuple[Metric, ...] = (*INTERVAL_METRICS, "activity", "wer")
IMPROVEMENT_METRICS = frozenset({"si_sdri", "overlap_si_sdri", "stoii"})


@dataclass(frozen=True)
class EvaluationConfig:
    """Resolved score/report settings, package versions, and asset content hashes.

    Paths and selection filters belong outside this configuration.

    Attributes:
        metrics: Selected metric names, sorted and deduplicated during construction.
        parameters: Resolved metric, VAD, speaker, and ASR backend settings,
            including defaults supplied by the caller.
        metric_backend_versions: Mapping of metric library names to versions.
        vad_backend_version: Silero VAD version, or None when unused.
        speaker_backend_version: Speaker extractor version, or None when unused.
        asr_backend_version: pywhispercpp version, or None when unused.
        model_digests: Mapping of logical asset names to content hashes.
        sample_rate: Evaluation sample rate in Hz, fixed at 16000 by the protocol.
        package_version: RTSS release that produced the results.
    """

    metrics: tuple[Metric, ...] = ALL_METRICS
    parameters: dict[str, object] = field(default_factory=dict)
    metric_backend_versions: dict[str, str] = field(default_factory=dict)
    vad_backend_version: str | None = None
    speaker_backend_version: str | None = None
    asr_backend_version: str | None = None
    model_digests: dict[str, str] = field(default_factory=dict)
    sample_rate: Literal[16000] = 16000
    package_version: str = __version__

    def __post_init__(self) -> None:
        """Reject unsupported settings and canonicalize metric selection order."""
        if not set(self.metrics) <= set(ALL_METRICS):
            raise ValueError("Unknown evaluation metric")
        if self.sample_rate != 16000:
            raise ValueError("The evaluation protocol requires 16000 Hz")
        object.__setattr__(self, "metrics", tuple(sorted(set(self.metrics))))


@dataclass(frozen=True)
class Interval:
    """Half-open sample bounds, shared directly by speech regions and results.

    Raw VAD intervals are active across their entire span. Merged utterances and
    overlap regions keep their original activity separately in SpeechRegion.

    Attributes:
        start: Inclusive sample index at the evaluation rate.
        end: Exclusive sample index at the evaluation rate.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject empty, reversed, or negative sample bounds."""
        if not 0 <= self.start < self.end:
            raise ValueError("Invalid interval bounds")


@dataclass(frozen=True)
class SpeechRegion:
    """An utterance or overlap span after merging/padding.

    Attributes:
        interval: Sample bounds of the region after merging/padding.
        active_samples: Raw active samples, excluding merged silent gaps and added
            overlap padding. Used for duration eligibility and not repeated in
            raw VAD or metric results.
    """

    interval: Interval
    active_samples: int

    def __post_init__(self) -> None:
        """Reject activity outside the region's span."""
        if not 0 <= self.active_samples <= self.interval.end - self.interval.start:
            raise ValueError("Invalid active duration")


@dataclass(frozen=True)
class AudioSource:
    """Relative audio path, content hash, and native metadata when readable.

    Attributes:
        path: Audio file path relative to the evaluation input directory.
        sha256: File content hash for result reuse, or None when unavailable.
        native_sample_rate: Original file sample rate in Hz, or None when unknown.
        native_channels: Original file channel count, or None when unknown.
    """

    path: str
    sha256: str | None
    native_sample_rate: int | None = None
    native_channels: int | None = None


@dataclass(frozen=True)
class AudioSources:
    """The three supplied files; missing/unreadable files retain their paths.

    Attributes:
        mixture: Input mixture file and its metadata.
        reference: Stereo reference file and its metadata.
        separated: Separator output file and its metadata.
    """

    mixture: AudioSource
    reference: AudioSource
    separated: AudioSource


@dataclass(frozen=True)
class IntervalMetricResult:
    """One metric candidate, with paired absolute scores for improvements.

    Attributes:
        interval: Sample bounds used for this metric candidate.
        interval_source: How the bounds were selected: reference VAD utterances
            for intrusive metrics, reference overlap for overlap SI-SDRi, and
            separated VAD utterances for DNSMOS and speaker similarity. Intrusive
            metrics still compare both waveforms over identical bounds; speaker
            similarity compares a separated embedding with the reference prototype.
        status: Outcome for this candidate. Examples:
            scored: A valid STOI value.
            ineligible: A reference interval with too little speech.
            undefined: SI-SDR with zero-energy output after subtracting its mean.
            failed: A metric backend exception or nonfinite score.
        value: Metric score, or None for non-scored results. For SI-SDRi and STOIi,
            value = output_score - input_score. For example, -2 dB input and 4 dB
            output give value=6 dB. Absolute metrics such as PESQ use value alone.
        reason: Explanation required for non-scored results.
        input_score: Mixture baseline for an improvement metric; for example,
            SI-SDR(mixture, reference).
        output_score: Separated score for an improvement metric; for example,
            SI-SDR(separation, reference). A finite score may survive an
            unavailable improvement baseline.
    """

    interval: Interval
    interval_source: IntervalSource
    status: IntervalResultStatus
    value: float | None = None
    reason: str | None = None
    input_score: float | None = None
    output_score: float | None = None

    def __post_init__(self) -> None:
        """Preserve valid paired components while rejecting nonfinite scores."""
        for name in ("value", "input_score", "output_score"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                object.__setattr__(self, name, None)
                object.__setattr__(self, "status", "failed")
                object.__setattr__(self, "reason", "nonfinite_score")
        if self.status == "failed":
            object.__setattr__(self, "value", None)
        if self.status == "scored":
            if self.value is None or self.reason is not None:
                raise ValueError("Scored interval results require a value and no failure reason")
        elif self.value is not None or not self.reason:
            raise ValueError("Non-scored interval results require a null value and a reason")


@dataclass(frozen=True)
class MetricResult:
    """Availability and the complete interval candidate population for one metric.

    Attributes:
        status: Availability of this metric:
            enabled: Selected on an available output, even if VAD found no candidates.
            disabled: Not requested.
            unavailable: The output channel is missing.
        interval_results: Individual candidate successes and errors.
        reason: Explanation required when the metric is unavailable.
    """

    status: Literal["enabled", "disabled", "unavailable"] = "disabled"
    interval_results: tuple[IntervalMetricResult, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        """Keep disabled and unavailable metrics out of candidate populations."""
        if self.status != "enabled" and self.interval_results:
            raise ValueError("Only enabled metrics can contain interval results")
        if self.status == "unavailable" and not self.reason:
            raise ValueError("Unavailable metrics require a reason")


@dataclass(frozen=True)
class ActivityCounts:
    """Raw 100 ms frame counts.

    Attributes:
        tp: Frames where both reference and separated speech are active.
        fp: Frames where separated speech is active and reference speech is inactive.
        fn: Frames where reference speech is active and separated speech is inactive.
        frames: Total frame count, including true negatives.
    """

    tp: int
    fp: int
    fn: int
    frames: int

    def __post_init__(self) -> None:
        """Reject negative or inconsistent confusion counts."""
        if min(self.tp, self.fp, self.fn, self.frames) < 0 or self.tp + self.fp + self.fn > self.frames:
            raise ValueError("Invalid activity counts")


@dataclass(frozen=True)
class ActivityResult:
    """One channel's full-recording speech-activity outcome.

    Attributes:
        status: Outcome of the activity comparison:
            scored: VAD comparison produced frame counts, including all-zero counts.
            disabled: Activity was not requested.
            unavailable: The separated channel is missing.
            failed: The VAD backend could not produce activity.
        counts: Frame counts present only for scored outcomes.
        reason: Explanation required for unavailable or failed outcomes.
    """

    status: CountResultStatus = "disabled"
    counts: ActivityCounts | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Require counts exactly when activity was scored."""
        _validate_count_result(self.status, self.counts, self.reason)


@dataclass(frozen=True)
class WordCounts:
    """Full-channel ASR agreement edits, including empty-reference insertions.

    Attributes:
        substitutions: Reference words replaced by different separated-transcript words.
        deletions: Reference words missing from the separated transcript.
        insertions: Extra words in the separated transcript, including empty-reference cases.
        reference_words: Total reference words, including hits, substitutions, and deletions.
    """

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    def __post_init__(self) -> None:
        """Reject impossible edit counts without imposing a WER upper bound."""
        if min(self.substitutions, self.deletions, self.insertions, self.reference_words) < 0:
            raise ValueError("Negative word counts")
        if self.substitutions + self.deletions > self.reference_words:
            raise ValueError("Substitutions and deletions exceed reference words")


@dataclass(frozen=True)
class Transcript:
    """Raw and normalized full-channel text with the resolved ASR language.

    Attributes:
        raw: Original text returned by the ASR backend.
        normalized: Text after the protocol's shared WER normalization.
        language: Resolved ASR language code, or None when unavailable.
    """

    raw: str
    normalized: str
    language: str | None = None


@dataclass(frozen=True)
class WordErrorResult:
    """One channel's full-recording transcription comparison.

    Recognizer/normalizer identities and settings live in config.

    Attributes:
        status: Outcome of the transcription comparison:
            scored: Edit counts were computed, including empty-reference cases
                whose WER ratio is undefined.
            disabled: WER was not requested.
            unavailable: The separated channel is missing.
            failed: The ASR backend or edit-count calculation failed.
        counts: Word edit counts present only for scored outcomes.
        reference: Reference-channel transcript, or None when unavailable.
        separated: Separated-channel transcript, or None when unavailable.
        reason: Explanation required for unavailable or failed outcomes.
    """

    status: CountResultStatus = "disabled"
    counts: WordCounts | None = None
    reference: Transcript | None = None
    separated: Transcript | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Require edit counts and both transcripts for a scored comparison."""
        _validate_count_result(self.status, self.counts, self.reason)
        if self.status == "scored" and (self.reference is None or self.separated is None):
            raise ValueError("Scored WER requires both transcripts")


def _validate_count_result(status: CountResultStatus, counts: object, reason: str | None) -> None:
    """Validate the common activity/transcription availability contract."""
    if (status == "scored") != (counts is not None):
        raise ValueError("Only scored count results must contain counts")
    if status in ("unavailable", "failed") and not reason:
        raise ValueError("Unavailable/failed count results require a reason")


@dataclass(frozen=True)
class ReferenceEmbeddingExtractionResult:
    """Reporting outcome for one reference-prototype candidate, without its vector.

    Actual vectors belong to the speaker adapter's in-memory state for prototype
    construction and cosine scoring. Eligibility/failure outcomes suffice for coverage.

    Attributes:
        interval: Reference utterance bounds used for the extraction candidate.
        status: Outcome of the reference embedding extraction:
            scored: Extraction succeeded.
            ineligible: Less than three seconds of raw active speech.
            undefined: A zero-norm embedding.
            failed: The speaker backend raised an extraction error.
        reason: Explanation required when extraction did not succeed.
    """

    interval: Interval
    status: IntervalResultStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        """Require a reason when extraction did not succeed."""
        if self.status != "scored" and not self.reason:
            raise ValueError("Unscored reference embeddings require a reason")


@dataclass(frozen=True)
class ReferencePrototypeExtractionResult:
    """Reference prototype extraction outcome for reporting.

    Its vector remains in the speaker adapter's in-memory state; this record
    preserves extraction outcomes for reporting.

    Attributes:
        status: Outcome of reference prototype extraction:
            available: All required extractions produced a usable mean vector.
            unavailable: No eligible utterances or a zero-norm prototype.
            disabled: Speaker similarity was not requested.
            failed: A required extraction failed in the speaker backend.
        extractions: Eligibility and extraction outcomes for reference utterance candidates.
        reason: Explanation required when the prototype is unavailable or failed.
    """

    status: Literal["available", "unavailable", "disabled", "failed"] = "disabled"
    extractions: tuple[ReferenceEmbeddingExtractionResult, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        """A required extraction failure must not leave an available prototype."""
        if self.status in ("unavailable", "failed") and not self.reason:
            raise ValueError("Unavailable/failed prototypes require a reason")
        if self.status == "disabled" and self.extractions:
            raise ValueError("Disabled prototypes cannot contain extractions")
        if self.status == "available" and (
            not any(item.status == "scored" for item in self.extractions)
            or any(item.status in ("failed", "undefined") for item in self.extractions)
        ):
            raise ValueError("Available prototypes require successful eligible extractions")


@dataclass(frozen=True)
class ChannelResult:
    """All channel outcomes, including reference data for missing residual output.

    Attributes:
        status: available when the separated WAV contains this channel, or
            unavailable for a missing residual channel in target-only output.
        reference_active: Activity from raw VAD, independent of utterance eligibility.
            False means no speech was detected anywhere in this reference channel;
            None means VAD was not obtained.
        reference_vad: Raw speech intervals detected across the full reference channel.
        separated_vad: Raw speech intervals detected across the full separated channel.
        reference_utterances: Reference speech regions after merging and duration filtering.
        separated_utterances: Separated speech regions after merging and duration filtering.
        separated_missing_utterances: Reference utterance intervals whose separated
            waveform has zero energy after subtracting that interval's arithmetic
            mean (centering for SI-SDR's zero_mean=True). Quiet, varying separated
            speech with no detected VAD activity does not meet this condition and
            is still scored.
        metrics: Interval metric outcomes keyed by metric name, including availability.
        activity: Full-channel speech-activity comparison and frame counts.
        wer: Full-channel transcription comparison and word edit counts.
        prototype: Reference speaker prototype extraction outcome for this channel.
    """

    status: Literal["available", "unavailable"] = "available"
    reference_active: bool | None = None
    reference_vad: tuple[Interval, ...] = ()
    separated_vad: tuple[Interval, ...] = ()
    reference_utterances: tuple[SpeechRegion, ...] = ()
    separated_utterances: tuple[SpeechRegion, ...] = ()
    separated_missing_utterances: tuple[Interval, ...] = ()
    metrics: dict[str, MetricResult] = field(
        default_factory=lambda: {name: MetricResult() for name in INTERVAL_METRICS}
    )
    activity: ActivityResult = field(default_factory=ActivityResult)
    wer: WordErrorResult = field(default_factory=WordErrorResult)
    prototype: ReferencePrototypeExtractionResult = field(default_factory=ReferencePrototypeExtractionResult)

    def __post_init__(self) -> None:
        """Require explicit statuses for every interval metric."""
        if set(self.metrics) != set(INTERVAL_METRICS):
            raise ValueError("Channel results require all interval metric statuses")
        for name in IMPROVEMENT_METRICS:
            for result in self.metrics[name].interval_results:
                if result.status == "scored":
                    if result.input_score is None or result.output_score is None:
                        raise ValueError("Scored improvements require paired input/output scores")
                    if not math.isclose(result.value, result.output_score - result.input_score):
                        raise ValueError("Improvement does not match its paired scores")


@dataclass(frozen=True)
class RecordingError:
    """A loading/validation error at the audio boundary, preserving its path.

    Attributes:
        path: Path of the audio file that could not be loaded or validated.
        reason: Description of the loading or waveform-validation failure.
    """

    path: str
    reason: str


@dataclass(frozen=True)
class RecordingResult:
    """One separation's result, with package-version provenance in configuration.

    Summaries also count processed recordings containing any failed metric as
    failed recordings. Which results were loaded from disk during an invocation
    is tracked separately through cached_result_paths in reporting.

    Attributes:
        inference_hash: Identifier of the separator configuration used for this output.
        evaluation_id: Hash identifying the resolved evaluation configuration.
        configuration: Evaluation settings, backend versions, model hashes, and package version.
        sources: Mixture, reference, and separated file paths, hashes, and native metadata.
        sample_count: Aligned recording length at the evaluation rate, or None when unknown.
        channels: Both T and R outcomes, including an unavailable R for target-only output.
        status: Outcome of recording evaluation:
            processed: Valid audio reached evaluation, even if individual metrics failed.
            invalid: Loading or validation failed, such as a missing reference WAV
                or unequal lengths.
            failed: Recording-wide evaluation could not complete, such as VAD
                failing before candidate regions could be constructed.
        error: Loading or validation failure required for invalid recordings; otherwise None.
        reference_overlap: Reference overlap regions after merging, filtering, and padding.
        output_channels: Read-only tuple of available separated channels in T/R order,
            derived from channel availability and omitted from JSON.
    """

    inference_hash: str
    evaluation_id: str
    configuration: EvaluationConfig
    sources: AudioSources
    sample_count: int | None
    channels: dict[Channel, ChannelResult]
    status: Literal["processed", "invalid", "failed"] = "processed"
    error: RecordingError | None = None
    reference_overlap: tuple[SpeechRegion, ...] = ()

    @property
    def output_channels(self) -> tuple[Channel, ...]:
        """Derive present separated channels in T/R order; omit this property in JSON."""
        return tuple(channel for channel in ("T", "R") if self.channels[channel].status == "available")

    def __post_init__(self) -> None:
        """Reject incomplete channel layouts and invalid recording outcomes."""
        if set(self.channels) != {"T", "R"}:
            raise ValueError("Results require both T and R channel records")
        if self.status == "invalid":
            if self.error is None:
                raise ValueError("Invalid recordings require a recording error")
        elif self.error is not None:
            raise ValueError("Only invalid recordings contain a recording error")
        if self.status != "invalid":
            if self.sample_count is None or self.sample_count <= 0 or self.output_channels not in (("T",), ("T", "R")):
                raise ValueError("Evaluated recordings require samples and T or T/R outputs")
            for result in self.channels.values():
                outcomes = {**result.metrics, "activity": result.activity, "wer": result.wer}
                for name, outcome in outcomes.items():
                    if name not in self.configuration.metrics:
                        if outcome.status != "disabled":
                            raise ValueError("Unselected metrics must be explicitly disabled")
                    elif outcome.status == "disabled":
                        raise ValueError("Selected metrics require explicit outcomes")
                    elif result.status == "unavailable" and outcome.status != "unavailable":
                        raise ValueError("Missing output channels require unavailable outcomes")
                    elif result.status == "available" and outcome.status == "unavailable":
                        raise ValueError("Selected metrics on available outputs require scoring outcomes")
                if "speaker_cosine" in self.configuration.metrics:
                    if result.prototype.status == "disabled":
                        raise ValueError("Speaker scoring requires a reference prototype outcome")
                elif result.prototype.status != "disabled":
                    raise ValueError("Unselected speaker scoring must disable prototype extraction")
