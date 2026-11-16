"""Offline reporting contracts using hand-built counts and interval results."""

import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest

from realtime_speech_separation import __version__
from realtime_speech_separation.evaluation.reporting import (
    compatible_result,
    content_digest,
    coverage,
    distribution,
    evaluation_identity,
    group_results,
    parse_result,
    selection_identity,
    strict_json_dumps,
    summarize_results,
    write_result,
)
from realtime_speech_separation.evaluation.types import (
    INTERVAL_METRICS,
    ActivityCounts,
    ActivityResult,
    AudioSource,
    AudioSources,
    ChannelResult,
    EvaluationConfig,
    Interval,
    IntervalMetricResult,
    MetricResult,
    RecordingError,
    RecordingResult,
    ReferenceEmbeddingExtractionResult,
    ReferencePrototypeExtractionResult,
    SpeechRegion,
    Transcript,
    WordCounts,
    WordErrorResult,
)


def channel_result(
    configuration: EvaluationConfig,
    *,
    metrics: dict[str, tuple[IntervalMetricResult, ...]] | None = None,
    available: bool = True,
    activity: ActivityResult | None = None,
    wer: WordErrorResult | None = None,
    prototype: ReferencePrototypeExtractionResult | None = None,
    **kwargs,
) -> ChannelResult:
    """Build explicit disabled/unavailable statuses around each test's measurements."""
    outcomes = {name: MetricResult() for name in INTERVAL_METRICS}
    for name in INTERVAL_METRICS:
        if name in configuration.metrics:
            outcomes[name] = (
                MetricResult("enabled", (metrics or {}).get(name, ()))
                if available
                else MetricResult("unavailable", reason="missing_output_channel")
            )
    return ChannelResult(
        status="available" if available else "unavailable",
        metrics=outcomes,
        activity=activity or ActivityResult(),
        wer=wer or WordErrorResult(),
        prototype=prototype or ReferencePrototypeExtractionResult(),
        **kwargs,
    )


def recording(
    name: str = "a",
    *,
    configuration: EvaluationConfig | None = None,
    target: ChannelResult | None = None,
    residual: ChannelResult | None = None,
    target_only: bool = False,
    sample_count: int = 16000,
    **kwargs,
) -> RecordingResult:
    """Supply a complete result envelope without loading audio or metric backends."""
    configuration = configuration or EvaluationConfig(metrics=("stoi",))
    return RecordingResult(
        inference_hash="inference_with_suffix",
        evaluation_id=evaluation_identity(configuration),
        configuration=configuration,
        sources=AudioSources(
            AudioSource(f"dataset/{name}.wav", "mixture-hash", 48000, 1),
            AudioSource(f"dataset/{name}_reference.wav", "reference-hash", 16000, 2),
            AudioSource(
                f"dataset/{name}_separated_inference_with_suffix.wav", "separation-hash", 16000, 1 if target_only else 2
            ),
        ),
        sample_count=sample_count,
        channels={
            "T": target or channel_result(configuration),
            "R": residual or channel_result(configuration, available=not target_only),
        },
        **kwargs,
    )


def scored(value: float, interval: Interval | None = None, **kwargs) -> IntervalMetricResult:
    """Construct one reference-utterance score."""
    return IntervalMetricResult(interval or Interval(0, 8000), "reference_utterance", "scored", value, **kwargs)


def summary(*results: RecordingResult, **kwargs) -> dict:
    """Summarize a nonempty identity group with test-specific run provenance."""
    return summarize_results(
        results, configuration=results[0].configuration, inference_hash=results[0].inference_hash, **kwargs
    )


def word_result(counts: WordCounts) -> WordErrorResult:
    """Preserve raw/normalized transcript fields alongside synthetic edit counts."""
    return WordErrorResult(
        "scored", counts, Transcript("Reference!", "reference", "en"), Transcript("Estimate", "estimate", "en")
    )


def test_pool_utterances_instead_of_recording_means() -> None:
    config = EvaluationConfig(metrics=("stoi",))
    first = recording("a", target=channel_result(config, metrics={"stoi": (scored(0),)}))
    second = recording(
        "b",
        target=channel_result(
            config, metrics={"stoi": (scored(1), scored(1, Interval(8000, 16000)), scored(1, Interval(16000, 24000)))}
        ),
        sample_count=24000,
    )

    result = summary(first, second)["channels"]["T"]["metrics"]["stoi"]

    assert result["distribution"]["count"] == 4
    assert result["distribution"]["mean"] == 0.75
    assert result["distribution"]["p10"] == pytest.approx(0.3)
    assert result["distribution"]["q1"] == 0.75
    assert result["coverage"]["scored_per_candidate"] == 1


def test_partial_failures_and_coverage_do_not_discard_other_metrics_or_channels() -> None:
    config = EvaluationConfig(metrics=("stoi", "pesq"))
    candidates = (
        scored(0),
        IntervalMetricResult(Interval(8000, 12000), "reference_utterance", "ineligible", reason="too_short"),
        IntervalMetricResult(Interval(12000, 20000), "reference_utterance", "undefined", reason="zero_energy"),
        IntervalMetricResult(Interval(20000, 28000), "reference_utterance", "failed", reason="metric_backend_error"),
    )
    record = recording(
        configuration=config,
        target=channel_result(config, metrics={"stoi": candidates, "pesq": (scored(3),)}),
        residual=channel_result(config, metrics={"stoi": (scored(0.8),)}),
        sample_count=28000,
    )
    result = summary(record)
    target = result["channels"]["T"]["metrics"]

    assert target["stoi"]["distribution"]["mean"] == 0
    assert target["pesq"]["distribution"]["mean"] == 3
    assert result["channels"]["R"]["metrics"]["stoi"]["distribution"]["mean"] == 0.8
    assert target["stoi"]["coverage"] == {
        "candidates": 4,
        "eligible": 3,
        "scored": 1,
        "ineligible": 1,
        "undefined": 1,
        "failed": 1,
        "scored_per_candidate": 0.25,
        "scored_per_eligible": 1 / 3,
        "reasons": {
            "ineligible": {"too_short": 1},
            "undefined": {"zero_energy": 1},
            "failed": {"metric_backend_error": 1},
        },
    }
    assert result["recordings"]["failed"] == 1


def test_improvements_pool_paired_differences_and_keep_unpaired_output_scores() -> None:
    config = EvaluationConfig(metrics=("si_sdri", "overlap_si_sdri"))
    target = channel_result(
        config,
        metrics={
            "si_sdri": (
                scored(2, input_score=-1, output_score=1),
                scored(6, Interval(8000, 16000), input_score=2, output_score=8),
                IntervalMetricResult(
                    Interval(16000, 24000),
                    "reference_utterance",
                    "undefined",
                    reason="zero_energy_baseline",
                    output_score=12,
                ),
            ),
            "overlap_si_sdri": (
                IntervalMetricResult(
                    Interval(4000, 12000), "reference_overlap", "scored", 8, input_score=1, output_score=9
                ),
            ),
        },
    )
    result = summary(recording(configuration=config, target=target, sample_count=24000))["channels"]["T"]["metrics"]

    assert result["si_sdri"]["distribution"]["mean"] == 4
    assert result["si_sdri"]["input_scores"]["count"] == 2
    assert result["si_sdri"]["output_scores"]["mean"] == 7
    assert result["overlap_si_sdri"]["distribution"]["mean"] == 8
    with pytest.raises(ValueError, match="paired"):
        channel_result(config, metrics={"si_sdri": (scored(2),)})
    with pytest.raises(ValueError, match="match"):
        channel_result(config, metrics={"si_sdri": (scored(2, input_score=1, output_score=4),)})


def test_activity_and_wer_use_summed_counts_including_empty_reference_insertions() -> None:
    config = EvaluationConfig(metrics=("activity", "wer"))
    first = recording(
        "a",
        configuration=config,
        target=channel_result(
            config,
            activity=ActivityResult("scored", ActivityCounts(1, 0, 0, 10)),
            wer=word_result(WordCounts(0, 1, 0, 1)),
        ),
        residual=channel_result(
            config,
            available=False,
            activity=ActivityResult("unavailable", reason="missing_output_channel"),
            wer=WordErrorResult("unavailable", reason="missing_output_channel"),
        ),
        target_only=True,
    )
    second = recording(
        "b",
        configuration=config,
        target=channel_result(
            config,
            activity=ActivityResult("scored", ActivityCounts(1, 8, 0, 10)),
            wer=word_result(WordCounts(1, 0, 0, 9)),
        ),
        residual=first.channels["R"],
        target_only=True,
    )
    third = recording(
        "c",
        configuration=config,
        target=channel_result(
            config,
            activity=ActivityResult("scored", ActivityCounts(0, 2, 0, 10)),
            wer=word_result(WordCounts(0, 0, 3, 0)),
            reference_active=False,
        ),
        residual=first.channels["R"],
        target_only=True,
    )

    target = summary(first, second, third)["channels"]["T"]

    assert target["activity"]["precision"] == 2 / 12
    assert target["activity"]["recall"] == 1
    assert target["activity"]["f1"] == 4 / 14
    assert target["activity"]["coverage"]["scored_frames"] == 30
    assert target["activity"]["coverage"]["candidate_frames"] == 30
    assert target["wer"]["wer"] == 0.5
    assert target["wer"]["insertions"] == 3
    assert target["wer"]["coverage"]["empty_reference_recordings"] == 1
    assert target["inactive_reference_channels"] == 1
    assert target["active_output_frames_on_inactive_reference"] == 2
    assert summary(third)["channels"]["T"]["wer"]["wer"] is None


def test_zero_denominators_empty_distributions_and_missing_residuals() -> None:
    config = EvaluationConfig(metrics=("stoi", "activity", "wer"))
    target = channel_result(
        config, activity=ActivityResult("scored", ActivityCounts(0, 0, 0, 10)), wer=word_result(WordCounts(0, 0, 0, 0))
    )
    residual = channel_result(
        config,
        available=False,
        activity=ActivityResult("unavailable", reason="missing_output_channel"),
        wer=WordErrorResult("unavailable", reason="missing_output_channel"),
    )
    result = summary(recording(configuration=config, target=target, residual=residual, target_only=True))

    assert result["recordings"]["target_only"] == 1
    assert result["channels"]["R"]["metrics"]["stoi"]["recordings_by_status"]["unavailable"] == 1
    assert result["channels"]["T"]["wer"]["wer"] is None
    assert all(result["channels"]["T"]["activity"][name] is None for name in ("precision", "recall", "f1"))
    assert result["channels"]["T"]["metrics"]["stoi"]["coverage"]["scored_per_candidate"] is None
    assert result["channels"]["R"]["metrics"]["stoi"]["distribution"]["mean"] is None
    assert json.loads(strict_json_dumps(result))


def test_activity_and_wer_failures_preserve_other_scores_and_reduce_coverage() -> None:
    config = EvaluationConfig(metrics=("activity", "wer", "stoi"))
    target = channel_result(
        config,
        metrics={"stoi": (scored(0.8),)},
        activity=ActivityResult("failed", reason="vad_failed"),
        wer=WordErrorResult("failed", reason="transcription_failed"),
    )
    residual = channel_result(
        config,
        activity=ActivityResult("scored", ActivityCounts(1, 0, 0, 2)),
        wer=word_result(WordCounts(0, 0, 0, 1)),
    )
    record = recording(configuration=config, target=target, residual=residual, sample_count=1601)
    result = summary(record)

    assert result["channels"]["T"]["activity"]["coverage"] == {
        "candidate_recordings": 1,
        "scored_recordings": 0,
        "scored_per_candidate": 0,
        "candidate_frames": 2,
        "scored_frames": 0,
        "scored_per_candidate_frame": 0,
    }
    assert result["channels"]["T"]["wer"]["coverage"]["scored_per_candidate"] == 0
    assert result["channels"]["T"]["metrics"]["stoi"]["distribution"]["mean"] == 0.8
    assert result["channels"]["R"]["wer"]["wer"] == 0
    assert result["recordings"]["failed"] == 1
    assert reuse(strict_json_dumps(record), record) is None


def test_prototype_coverage_and_silent_diagnostics_are_separate() -> None:
    config = EvaluationConfig(metrics=("speaker_cosine",))
    reference = Interval(0, 48000)
    missing = Interval(48000, 56000)
    separated = Interval(0, 48000)
    target = channel_result(
        config,
        prototype=ReferencePrototypeExtractionResult(
            "available",
            (
                ReferenceEmbeddingExtractionResult(reference, "scored"),
                ReferenceEmbeddingExtractionResult(missing, "ineligible", "too_short"),
            ),
        ),
        reference_utterances=(SpeechRegion(reference, 48000), SpeechRegion(missing, 8000)),
        separated_utterances=(SpeechRegion(separated, 48000),),
        metrics={"speaker_cosine": (IntervalMetricResult(separated, "separated_utterance", "scored", -0.25),)},
        separated_missing_utterances=(missing,),
    )
    residual = channel_result(
        config, prototype=ReferencePrototypeExtractionResult("unavailable", reason="no_reference_utterances")
    )
    record = recording(configuration=config, target=target, residual=residual, sample_count=56000)
    result = summary(record)["channels"]["T"]

    assert result["prototype"]["coverage"]["candidates"] == 2
    assert result["metrics"]["speaker_cosine"]["coverage"]["candidates"] == 1
    assert result["metrics"]["speaker_cosine"]["distribution"]["mean"] == -0.25
    assert result["separated_missing_utterance_count"] == 1
    assert target.prototype.extractions[0].interval is reference
    assert target.metrics["speaker_cosine"].interval_results[0].interval is separated
    assert target.separated_missing_utterances[0] is missing
    assert parse_result(strict_json_dumps(record)) == record
    with pytest.raises(ValueError, match="eligible extractions"):
        ReferencePrototypeExtractionResult(
            "available",
            (
                ReferenceEmbeddingExtractionResult(reference, "scored"),
                ReferenceEmbeddingExtractionResult(missing, "failed", "extraction_error"),
            ),
        )


def test_empty_and_singleton_distributions() -> None:
    assert distribution([]) == {
        "count": 0,
        **dict.fromkeys(("mean", "sd", "min", "p10", "q1", "median", "q3", "p90", "max")),
    }
    assert distribution([2]) == {
        "count": 1,
        "sd": 0,
        **dict.fromkeys(("mean", "min", "p10", "q1", "median", "q3", "p90", "max"), 2),
    }
    assert coverage([])["scored_per_eligible"] is None
    assert (
        coverage([IntervalMetricResult(Interval(0, 4000), "reference_utterance", "ineligible", reason="too_short")])[
            "eligible"
        ]
        == 0
    )


def test_empty_and_all_invalid_groups_have_strict_zero_count_summaries() -> None:
    config = EvaluationConfig(metrics=("stoi",))
    empty = summarize_results([], configuration=config, inference_hash="known", selected_mixtures=("dataset/a.wav",))
    assert empty["recordings"]["total"] == 0
    assert empty["selection"]["mixture_paths"] == ["dataset/a.wav"]
    assert empty["channels"]["T"]["metrics"]["stoi"]["distribution"]["mean"] is None
    invalid = recording(status="invalid", error=RecordingError("dataset/a.wav", "unreadable"))
    result = summary(invalid)
    assert result["recordings"] == {
        "total": 1,
        "processed": 0,
        "invalid": 1,
        "reused": 0,
        "target_only": 0,
        "failed": 0,
    }
    assert json.loads(strict_json_dumps(result))


@pytest.mark.parametrize("component", ("value", "input_score", "output_score"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_metric_backend_scores_become_failed_null_interval_results(component: str, value: float) -> None:
    interval_result = scored(2, **{component: value}) if component != "value" else scored(value)
    assert interval_result.status == "failed"
    assert interval_result.reason == "nonfinite_score"
    assert interval_result.value is None
    assert getattr(interval_result, component) is None
    assert json.loads(strict_json_dumps(interval_result))["value"] is None


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_strict_serialization_and_distribution_reject_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError):
        strict_json_dumps({"value": value})
    with pytest.raises(ValueError):
        distribution([value])


def test_json_round_trip_and_atomic_write(tmp_path: Path) -> None:
    config = EvaluationConfig(metrics=("stoi",), parameters={"unicode": "é", "nested": [None, True, 0.5]})
    interval = Interval(0, 8000)
    record = recording(
        configuration=config,
        target=channel_result(
            config,
            metrics={"stoi": (scored(0.5, interval),)},
            reference_vad=(Interval(0, 3000), Interval(5000, 8000)),
            reference_utterances=(SpeechRegion(interval, 6000),),
        ),
    )
    path = tmp_path / "result.json"
    write_result(path, record)

    assert parse_result(path.read_text()) == record
    data = json.loads(path.read_text())
    assert data["configuration"]["package_version"] == __version__
    assert "output_channels" not in data
    assert data["channels"]["T"]["reference_vad"] == [{"start": 0, "end": 3000}, {"start": 5000, "end": 8000}]
    stored_region = data["channels"]["T"]["reference_utterances"][0]
    stored_score = data["channels"]["T"]["metrics"]["stoi"]["interval_results"][0]
    assert stored_region == {"interval": {"start": 0, "end": 8000}, "active_samples": 6000}
    assert stored_score["interval"] == stored_region["interval"]
    assert stored_score["interval_source"] == "reference_utterance"
    assert set(tmp_path.iterdir()) == {path}
    write_result(path, replace(record, status="failed"))
    assert parse_result(path.read_text()).status == "failed"


def test_atomic_write_preserves_existing_file_on_replace_error(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "result.json"
    path.write_text("original")

    def fail_replace(*args) -> None:
        """Simulate a filesystem failure at the atomic replacement boundary."""
        raise OSError("replace failed")

    monkeypatch.setattr("realtime_speech_separation.evaluation.reporting.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        write_result(path, recording())
    assert path.read_text() == "original"
    assert set(tmp_path.iterdir()) == {path}


def test_identity_uses_semantics_versions_assets_and_package_version() -> None:
    config = EvaluationConfig(metrics=("stoi", "pesq"), parameters={"b": 2, "a": 1})
    assert evaluation_identity(config) == evaluation_identity(
        replace(config, metrics=("pesq", "stoi", "stoi"), parameters={"a": 1, "b": 2})
    )
    for change in (
        {"metrics": ("stoi",)},
        {"parameters": {"a": 2, "b": 2}},
        {"metric_backend_versions": {"numpy": "different"}},
        {"vad_backend_version": "different"},
        {"speaker_backend_version": "different"},
        {"asr_backend_version": "different"},
        {"model_digests": {"vad": "different"}},
        {"package_version": "next"},
    ):
        assert evaluation_identity(config) != evaluation_identity(replace(config, **change))
    assert selection_identity() == "all"
    assert selection_identity(("AMI", "fisher", "ami")) == selection_identity(("FISHER", "ami"))
    assert selection_identity(("ami",)) != selection_identity(("fisher",))


def test_content_digest_tracks_content_not_filename(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    assert content_digest(first) == content_digest(second)
    second.write_bytes(b"changed")
    assert content_digest(first) != content_digest(second)


def reuse(payload: str, record: RecordingResult, **kwargs) -> RecordingResult | None:
    """Attempt cache reuse with default validity keys from a result."""
    return compatible_result(
        payload,
        **{
            "configuration": record.configuration,
            "inference_hash": record.inference_hash,
            "sources": record.sources,
            **kwargs,
        },
    )


def test_compatible_result_checks_all_sources_and_configuration() -> None:
    record = recording()
    payload = strict_json_dumps(record)
    assert reuse(payload, record) == record
    assert reuse(payload, record, inference_hash="other") is None
    assert reuse(payload, record, configuration=replace(record.configuration, parameters={"changed": True})) is None
    for name in ("mixture", "reference", "separated"):
        source = getattr(record.sources, name)
        for changes in ({"sha256": "changed"}, {"sha256": None}, {"path": "other.wav"}):
            assert reuse(payload, record, sources=replace(record.sources, **{name: replace(source, **changes)})) is None
    # Native metadata is informational; content hashes establish actual source validity.
    assert (
        reuse(
            payload,
            record,
            sources=replace(record.sources, mixture=replace(record.sources.mixture, native_sample_rate=None)),
        )
        == record
    )


def test_published_package_version_is_recorded_and_controls_result_reuse() -> None:
    current = recording()
    previous_config = replace(current.configuration, package_version="previous-release")
    previous = replace(current, configuration=previous_config, evaluation_id=evaluation_identity(previous_config))
    payload = strict_json_dumps(previous)

    assert parse_result(payload).configuration.package_version == "previous-release"
    assert reuse(payload, current) is None
    assert reuse(payload, previous) == previous
    assert summary(current)["configuration"]["package_version"] == __version__
    assert summary(current)["percentile_method"] == "linear"
    assert "percentile_method" not in asdict(current.configuration)


def test_output_channels_is_derived_and_not_serialized() -> None:
    stereo = recording()
    mono = recording(target_only=True)

    assert stereo.output_channels == ("T", "R")
    assert mono.output_channels == ("T",)
    assert mono.channels.keys() == stereo.channels.keys() == {"T", "R"}
    assert parse_result(strict_json_dumps(mono)).output_channels == ("T",)
    with pytest.raises(FrozenInstanceError):
        mono.output_channels = ("T", "R")


@pytest.mark.parametrize("payload", ("", "{", "{}", "null", "[]", '{"si_sdr": [1, 2]}'))
def test_malformed_legacy_or_interrupted_results_are_not_reused(payload: str) -> None:
    assert reuse(payload, recording()) is None


@pytest.mark.parametrize(
    "mutation",
    (
        lambda data: data["configuration"].pop("package_version"),
        lambda data: data.update(evaluation_id="wrong"),
        lambda data: data.pop("channels"),
        lambda data: data["channels"]["T"].pop("wer"),
        lambda data: data["channels"]["T"]["metrics"].pop("stoi"),
        lambda data: data["channels"]["T"]["metrics"]["stoi"].update(status="disabled"),
        lambda data: data["channels"]["T"]["metrics"]["stoi"].update(
            status="unavailable", reason="missing_metric_backend"
        ),
        lambda data: data.update(sample_count=True),
        lambda data: data.update(sample_count=1.5),
        lambda data: data.update(status="unknown"),
        lambda data: data.update(extra="unknown"),
    ),
)
def test_structurally_incompatible_results_are_not_reused(mutation) -> None:
    record = recording()
    data = json.loads(strict_json_dumps(record))
    mutation(data)
    assert reuse(json.dumps(data), record) is None


def test_nonfinite_or_duplicate_json_is_not_reused() -> None:
    record = recording()
    payload = strict_json_dumps(record)
    assert reuse(payload.replace('"sample_count": 16000', '"sample_count": NaN'), record) is None
    assert (
        reuse(payload.replace('"sample_count": 16000', '"sample_count": 16000, "sample_count": 16000'), record) is None
    )
    with pytest.raises(ValueError):
        strict_json_dumps({1: "lossy key"})


def test_ineligible_and_undefined_results_are_reusable_but_partial_failures_are_not() -> None:
    config = EvaluationConfig(metrics=("stoi",))
    for status in ("ineligible", "undefined", "failed"):
        record = recording(
            target=channel_result(
                config,
                metrics={
                    "stoi": (IntervalMetricResult(Interval(0, 8000), "reference_utterance", status, reason="reason"),)
                },
            )
        )
        assert (reuse(strict_json_dumps(record), record) is not None) == (status != "failed")
    failed = recording(status="failed")
    invalid = recording(status="invalid", error=RecordingError("dataset/a.wav", "unreadable"))
    assert reuse(strict_json_dumps(failed), failed) is None
    assert reuse(strict_json_dumps(invalid), invalid) is None


def test_grouping_selection_and_cached_result_provenance() -> None:
    first, second = recording("a"), recording("b")
    changed_inference = replace(second, inference_hash="another")
    config = replace(second.configuration, parameters={"changed": True})
    changed_evaluation = replace(second, configuration=config, evaluation_id=evaluation_identity(config))
    assert len(group_results((first, second, changed_inference, changed_evaluation))) == 3
    before = asdict(first)
    report = summary(first, filter_terms=("DATASET", "dataset"), cached_result_paths=(first.sources.separated.path,))
    assert report["recordings"]["reused"] == 1
    assert report["recordings"]["processed"] == 1
    assert report["selection"]["filter_terms"] == ("dataset",)
    assert report["selection"]["mixture_paths"] == [first.sources.mixture.path]
    assert report["evaluation_id"] == first.evaluation_id
    assert asdict(first) == before
    for other in (changed_inference, changed_evaluation):
        with pytest.raises(ValueError, match="identities"):
            summary(first, other)
    with pytest.raises(ValueError, match="Duplicate"):
        summary(first, first)
    with pytest.raises(ValueError, match="selected"):
        summary(first, cached_result_paths=(second.sources.separated.path,))


@pytest.mark.parametrize(
    "make_invalid",
    (
        lambda: Interval(3, 2),
        lambda: Interval(0, 0),
        lambda: SpeechRegion(Interval(0, 10), 11),
        lambda: SpeechRegion(Interval(0, 10), -1),
        lambda: ActivityCounts(1, 1, 1, 2),
        lambda: WordCounts(2, 1, 0, 2),
        lambda: WordCounts(0, 0, -1, 0),
        lambda: ActivityResult("failed"),
        lambda: WordErrorResult("scored", WordCounts(0, 0, 0, 0)),
        lambda: IntervalMetricResult(Interval(0, 8000), "reference_utterance", "undefined"),
        lambda: IntervalMetricResult(Interval(0, 8000), "reference_utterance", "scored"),
        lambda: EvaluationConfig(metrics=("unknown",)),
        lambda: EvaluationConfig(sample_rate=8000),
    ),
)
def test_record_invariants_fail_early(make_invalid) -> None:
    with pytest.raises(ValueError):
        make_invalid()
