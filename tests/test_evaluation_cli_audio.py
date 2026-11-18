"""Synthetic end-to-end core CLI fixture with offline VAD and metric resources."""

import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from realtime_speech_separation.eval_cli import main
from realtime_speech_separation.evaluation import pipeline
from realtime_speech_separation.evaluation.metrics import MetricScore
from realtime_speech_separation.evaluation.reporting import parse_result, result_path
from realtime_speech_separation.evaluation.types import Interval
from realtime_speech_separation.evaluation.vad import (
    OVERLAP_MIN_ACTIVE_SAMPLES,
    UTTERANCE_MIN_ACTIVE_SAMPLES,
    VadBackendError,
)


@pytest.fixture
def core_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Inject resources at their preflight boundary, exercising the rest of the CLI."""
    state = SimpleNamespace(
        loads=Counter(),
        vad_calls=[],
        scores=[],
        metric_failure=False,
        vad_failure=None,
        separated_timing={},
        vad_intervals={},
        metric_parameters={},
        visqol_model_digest="model-a",
        dnsmos_model_digests={"dnsmos/model_v8.onnx": "p808-a", "dnsmos/sig_bak_ovr.onnx": "companion-a"},
        dnsmos_scores=[],
    )

    class Detector:
        """Return distinct reference T/R timing and matching output timing."""

        backend_version = "test-vad"
        torch_version = "test-torch"
        model_digest = "test-model"
        parameters = {"threshold": 0.5}

        def __init__(self):
            """Track once-per-invocation model initialization."""
            state.loads["vad"] += 1

        def __call__(self, samples):
            """Map synthetic channel identifiers to independent speech intervals."""
            identifier = round(float(samples[0]))
            state.vad_calls.append(identifier)
            failures = state.vad_failure if isinstance(state.vad_failure, tuple) else (state.vad_failure,)
            if identifier in failures:
                raise VadBackendError("test VAD failure")
            if identifier in state.vad_intervals:
                return state.vad_intervals[identifier]
            if identifier == 0:
                return ()
            default = (0, 10000) if identifier in (1, 3) else (6000, 16000)
            start, end = state.separated_timing.get(identifier, default)
            return (Interval(min(start, max(0, len(samples) - 3200)), min(end, len(samples))),)

    def adapter_factory(family):
        """Create deterministic functions with version/settings provenance."""
        state.loads[family] += 1

        def score(reference, estimate):
            """Retain actual slices so tests can inspect channel/interval identity."""
            state.scores.append((family, reference.copy(), estimate.copy()))
            if state.metric_failure and family == "si_sdr" and round(float(estimate[0])) == 3:
                return MetricScore("failed", reason="test_metric_failure")
            return MetricScore("scored", float(reference[0]) + 10 * float(estimate[0]))

        score.versions = {family: "test"}
        score.parameters = {"test": True}
        if family == "si_sdr":
            score.parameters["min_active_samples"] = OVERLAP_MIN_ACTIVE_SAMPLES
        score.parameters.update(state.metric_parameters.get(family, {}))
        if family == "visqol":
            score.model_digest = state.visqol_model_digest
        return score

    def dnsmos_factory():
        """Load a fake nonintrusive adapter that receives separated audio only."""
        state.loads["dnsmos_p808"] += 1

        def score(estimate):
            """Retain the waveform actually passed to DNSMOS."""
            state.dnsmos_scores.append(estimate.copy())
            return MetricScore("scored", float(estimate[0]))

        score.versions = {"onnxruntime": "test"}
        score.parameters = {"test": True, **state.metric_parameters.get("dnsmos_p808", {})}
        score.model_digests = state.dnsmos_model_digests.copy()
        return score

    monkeypatch.setattr(pipeline, "SileroVad", Detector)
    monkeypatch.setattr(pipeline, "SiSdr", lambda: adapter_factory("si_sdr"))
    monkeypatch.setattr(pipeline, "Stoi", lambda: adapter_factory("stoi"))
    monkeypatch.setattr(pipeline, "Pesq", lambda: adapter_factory("pesq"))
    monkeypatch.setattr(pipeline, "Visqol", lambda: adapter_factory("visqol"))
    monkeypatch.setattr(pipeline, "DnsmosP808", dnsmos_factory)
    return state


def write_triple(
    root: Path, name: str = "call_name", channels: int = 2, variants=("config_with_suffix",), sample_count: int = 16000
) -> Path:
    """Write float WAVs with identifiable T/R, mixture, and separated samples."""
    mixture_file = root / f"{name}.wav"
    mixture_file.parent.mkdir(parents=True, exist_ok=True)
    ramp = np.linspace(0, 0.2, sample_count, dtype=np.float32)
    reference = np.stack((1 + ramp, 2 + ramp), axis=1)
    separated = np.stack((3 + ramp, 4 + ramp), axis=1)[:, :channels]
    sf.write(mixture_file, 5 + ramp, 16000, subtype="FLOAT")
    sf.write(root / f"{name}_reference.wav", reference, 16000, subtype="FLOAT")
    for variant in variants:
        sf.write(root / f"{name}_separated_{variant}.wav", separated, 16000, subtype="FLOAT")
    return mixture_file


def results_in(root: Path) -> list:
    """Parse only adjacent result records, validating their persisted schema."""
    return [parse_result(path.read_text()) for path in sorted(root.rglob("*_results_*.json"))]


@pytest.mark.parametrize("channels", [1, 2])
def test_cli_loads_and_scores_triple_without_enrollment(tmp_path, core_runtime, channels, caplog) -> None:
    """The whole pipeline preserves T/R assignment and valid target-only results."""
    write_triple(tmp_path, channels=channels)
    assert main(["--input_dir", str(tmp_path), "--metrics", "core", "--overwrite"]) == 0
    (record,) = results_in(tmp_path)
    assert record.inference_hash == "config_with_suffix"
    assert record.output_channels == (("T",) if channels == 1 else ("T", "R"))
    assert record.sources.mixture.native_sample_rate == 16000
    assert record.sources.separated.native_channels == channels
    assert record.sources.reference.sha256
    assert record.configuration.parameters["protocol_scope"] == "partial"
    assert "partial protocol evaluation" in caplog.text
    assert not (tmp_path / "call_name_enrollment.wav").exists()
    target = record.channels["T"]
    assert target.metrics["si_sdri"].interval_results[0].interval == Interval(0, 10000)
    (overlap,) = target.metrics["overlap_si_sdri"].interval_results
    assert overlap.interval == Interval(4000, 12000)
    assert overlap.status == "scored"
    assert record.configuration.parameters["si_sdr"]["min_active_samples"] == OVERLAP_MIN_ACTIVE_SAMPLES
    assert record.configuration.parameters["stoi"]["min_active_samples"] == UTTERANCE_MIN_ACTIVE_SAMPLES
    assert target.activity.counts.tp == 6
    for _family, reference, estimate in core_runtime.scores:
        assert reference.dtype == estimate.dtype == np.float32
        assert reference.shape == estimate.shape
        expected = (3, 5) if round(float(reference[0])) == 1 else (4, 5)
        assert round(float(estimate[0])) in expected
    for channel in record.output_channels:
        for metric in ("si_sdri", "overlap_si_sdri", "stoii"):
            for score in record.channels[channel].metrics[metric].interval_results:
                assert score.value == pytest.approx(score.output_score - score.input_score)
    (summary_file,) = tmp_path.glob("summary_*.json")
    summary = json.loads(summary_file.read_text())
    assert summary["recordings"]["total"] == 1
    assert summary["channels"]["T"]["metrics"]["stoi"]["coverage"]["scored"] == 1
    (csv_file,) = tmp_path.glob("metrics_*.csv")
    with csv_file.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["T_stoi_status"] == "enabled"
    assert rows[0]["T_activity_tp"] == "6"
    recording_counts = (
        "candidate_recordings",
        "scored_recordings",
        "no_candidate_recordings",
        "excluded_due_to_errors",
        "partially_failed_recordings",
        "empty_reference_recordings",
    )
    assert not any(column.endswith(recording_counts) for column in rows[0])
    assert "T_activity_candidate_frames" in rows[0]
    assert "T_stoi_candidates" in rows[0]


def test_cli_reuses_reference_baselines_and_cached_results(tmp_path, core_runtime) -> None:
    """Reference VAD/baselines stay local to a mixture and cached reruns do no scoring."""
    write_triple(tmp_path, variants=("one", "two"))
    args = ["--input_dir", str(tmp_path), "--metrics", "core"]
    assert main(args) == 0
    assert core_runtime.loads == {"vad": 1, "si_sdr": 1, "stoi": 1}
    assert core_runtime.vad_calls == [1, 2, 3, 4, 3, 4]
    # Two channels, SI-SDR utterance + overlap and STOI utterance: six baselines.
    baselines = [call for call in core_runtime.scores if round(float(call[2][0])) == 5]
    assert len(baselines) == 6
    assert sorted({len(call[1]) for call in baselines}) == [8000, 10000]
    count = len(core_runtime.scores)
    saved = {path: path.read_bytes() for path in tmp_path.glob("*_results_*.json")}
    assert main(args) == 0
    assert len(core_runtime.scores) == count
    assert core_runtime.vad_calls == [1, 2, 3, 4, 3, 4]
    assert all(path.read_bytes() == payload for path, payload in saved.items())
    assert all(json.loads(path.read_text())["recordings"]["reused"] == 1 for path in tmp_path.glob("summary_*.json"))
    assert main([*args, "--overwrite"]) == 0
    assert len(core_runtime.scores) == 2 * count


def test_changed_audio_configuration_and_malformed_results_recompute(tmp_path, core_runtime) -> None:
    """Source bytes and resolved settings both control persisted result reuse."""
    mixture = write_triple(tmp_path)
    args = ["--input_dir", str(tmp_path), "--metrics", "stoi"]
    assert main(args) == 0
    (first,) = results_in(tmp_path)
    count = len(core_runtime.scores)
    sf.write(mixture, np.linspace(5, 6, 16000, dtype=np.float32), 16000, subtype="FLOAT")
    assert main(args) == 0
    assert len(core_runtime.scores) > count
    (second,) = results_in(tmp_path)
    assert second.evaluation_id == first.evaluation_id
    assert second.sources.mixture.sha256 != first.sources.mixture.sha256
    path = result_path(mixture, second.inference_hash, second.evaluation_id)
    path.write_text('{"legacy": true}')
    count = len(core_runtime.scores)
    assert main(args) == 0
    assert len(core_runtime.scores) > count
    assert main(["--input_dir", str(tmp_path), "--metrics", "activity"]) == 0
    assert len(results_in(tmp_path)) == 2


def test_filtered_cached_reports_coexist_with_full_set(tmp_path, core_runtime) -> None:
    """Only reached cached results contribute, preserving the earlier full-set report."""
    write_triple(tmp_path / "DatasetA")
    write_triple(tmp_path / "DatasetB", channels=1)
    args = ["--input_dir", str(tmp_path), "--metrics", "stoi"]
    assert main(args) == 0
    (full,) = tmp_path.glob("summary_*_all.json")
    saved = full.read_bytes()
    count = len(core_runtime.scores)
    assert main([*args, "--input_filter", "DATASETA"]) == 0
    assert len(core_runtime.scores) == count
    assert full.read_bytes() == saved
    (selected,) = [path for path in tmp_path.glob("summary_*.json") if path != full]
    summary = json.loads(selected.read_text())
    assert summary["recordings"]["total"] == summary["recordings"]["reused"] == 1
    assert summary["selection"]["filter_terms"] == ["dataseta"]
    assert summary["selection"]["mixture_paths"] == ["DatasetA/call_name.wav"]


@pytest.mark.parametrize("failure", ["metric", "reference_vad", "output_vad"])
def test_backend_failure_isolation_and_failed_result_retry(tmp_path, core_runtime, failure) -> None:
    """Failures return nonzero, preserve independent outcomes, and cannot be reused."""
    write_triple(tmp_path)
    if failure == "metric":
        core_runtime.metric_failure = True
    else:
        core_runtime.vad_failure = 1 if failure == "reference_vad" else 3
    args = ["--input_dir", str(tmp_path), "--metrics", "core"]
    assert main(args) == 1
    (record,) = results_in(tmp_path)
    if failure == "metric":
        assert record.channels["T"].metrics["si_sdri"].interval_results[0].status == "failed"
        assert record.channels["T"].metrics["stoi"].interval_results[0].status == "scored"
        assert record.channels["R"].metrics["si_sdri"].interval_results[0].status == "scored"
    elif failure == "output_vad":
        assert record.channels["T"].activity.status == "failed"
        assert record.channels["R"].activity.status == "scored"
        assert record.channels["T"].metrics["stoi"].interval_results[0].status == "scored"
    else:
        assert record.status == "failed"
        assert record.channels["T"].activity.status == "failed"
    core_runtime.metric_failure = False
    core_runtime.vad_failure = None
    assert main(args) == 0
    (summary_file,) = tmp_path.glob("summary_*.json")
    assert json.loads(summary_file.read_text())["recordings"]["reused"] == 0


def test_cli_continues_after_invalid_audio(tmp_path, core_runtime, caplog) -> None:
    """A missing reference becomes an invalid result without losing later inputs."""
    write_triple(tmp_path, "a_invalid")
    (tmp_path / "a_invalid_reference.wav").unlink()
    write_triple(tmp_path, "b_valid")
    assert main(["--input_dir", str(tmp_path), "--metrics", "core"]) == 1
    records = results_in(tmp_path)
    assert [record.status for record in records] == ["invalid", "processed"]
    assert records[0].error.path == "a_invalid_reference.wav"
    assert "a_invalid_reference.wav" in caplog.text
    (summary_file,) = tmp_path.glob("summary_*.json")
    summary = json.loads(summary_file.read_text())
    assert summary["recordings"]["invalid"] == summary["recordings"]["processed"] == 1


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("name", ["pesq", "visqol"])
def test_cli_reports_absolute_reference_utterance_scores(tmp_path, core_runtime, channels, name) -> None:
    """Score only the selected family, with matching reference and output bounds."""
    write_triple(tmp_path, channels=channels)
    assert main(["--input_dir", str(tmp_path), "--metrics", name]) == 0
    assert core_runtime.loads == {"vad": 1, name: 1}
    (record,) = results_in(tmp_path)
    assert record.configuration.metrics == (name,)
    assert record.configuration.metric_backend_versions[name] == "test"
    for channel, interval, value in (("T", Interval(0, 10000), 31), ("R", Interval(6000, 16000), 42.825)):
        metric = record.channels[channel].metrics[name]
        if channel not in record.output_channels:
            assert metric.status == "unavailable"
            assert not metric.interval_results
            continue
        (score,) = metric.interval_results
        assert score.interval == interval
        assert score.interval_source == "reference_utterance"
        assert score.value == pytest.approx(value, abs=0.001)
        assert score.input_score is score.output_score is None
    assert len(core_runtime.scores) == channels
    (summary_file,) = tmp_path.glob("summary_*.json")
    summary = json.loads(summary_file.read_text())
    assert summary["channels"]["T"]["metrics"][name]["distribution"]["mean"] == 31
    (csv_file,) = tmp_path.glob("metrics_*.csv")
    with csv_file.open(newline="") as stream:
        (row,) = csv.DictReader(stream)
    assert row[f"T_{name}_mean"] == "31.0"


def test_visqol_model_digest_controls_result_reuse(tmp_path, core_runtime) -> None:
    """The same local model reuses results and different model bytes change the evaluation ID."""
    write_triple(tmp_path, channels=1)
    args = ["--input_dir", str(tmp_path), "--metrics", "visqol"]
    assert main(args) == main(args) == 0
    assert len(core_runtime.scores) == 1
    core_runtime.visqol_model_digest = "model-b"
    assert main(args) == 0
    assert len(core_runtime.scores) == 2
    records = results_in(tmp_path)
    assert len({record.evaluation_id for record in records}) == 2
    assert {record.configuration.model_digests["visqol_speech.tflite"] for record in records} == {"model-a", "model-b"}


@pytest.mark.parametrize("channels", [1, 2])
def test_cli_dnsmos_uses_separated_utterance_bounds(tmp_path, core_runtime, channels) -> None:
    """Separated VAD independently determines the waveform slices and report bounds."""
    write_triple(tmp_path, channels=channels)
    core_runtime.separated_timing = {3: (2100, 10100), 4: (1000, 10000)}
    assert main(["--input_dir", str(tmp_path), "--metrics", "dnsmos_p808"]) == 0
    assert core_runtime.loads == {"vad": 1, "dnsmos_p808": 1}
    assert not core_runtime.scores
    (record,) = results_in(tmp_path)
    assert record.configuration.metric_backend_versions["onnxruntime"] == "test"
    assert record.configuration.model_digests.items() >= core_runtime.dnsmos_model_digests.items()
    separation = tmp_path / record.sources.separated.path
    waveform, _ = sf.read(separation, dtype="float32", always_2d=True)
    for index, (channel, (start, end)) in enumerate(
        zip(("T", "R"), core_runtime.separated_timing.values(), strict=True)
    ):
        result = record.channels[channel].metrics["dnsmos_p808"]
        if index >= channels:
            assert result.status == "unavailable"
            assert result.reason == "missing_output_channel"
            continue
        assert result.status == "enabled"
        (score,) = result.interval_results
        assert score.interval == Interval(start, end)
        assert score.interval_source == "separated_utterance"
        assert score.value == float(waveform[start, index])
        assert score.input_score is score.output_score is None
        np.testing.assert_array_equal(core_runtime.dnsmos_scores[index], waveform[start:end, index])
    assert len(core_runtime.dnsmos_scores) == channels
    (summary_file,) = tmp_path.glob("summary_*.json")
    summary = json.loads(summary_file.read_text())
    assert summary["channels"]["T"]["metrics"]["dnsmos_p808"]["coverage"]["scored"] == 1
    (csv_file,) = tmp_path.glob("metrics_*.csv")
    with csv_file.open(newline="") as stream:
        (row,) = csv.DictReader(stream)
    assert float(row["T_dnsmos_p808_mean"]) == float(waveform[2100, 0])


@pytest.mark.parametrize("failing_identifier", [1, 3])
def test_cli_dnsmos_preserves_scores_when_one_vad_fails(tmp_path, core_runtime, failing_identifier) -> None:
    """Reference VAD does not gate DNSMOS and separated VAD failures affect their own channel."""
    write_triple(tmp_path)
    core_runtime.vad_failure = failing_identifier
    assert main(["--input_dir", str(tmp_path), "--metrics", "dnsmos_p808"]) == 1
    (record,) = results_in(tmp_path)
    target = record.channels["T"].metrics["dnsmos_p808"]
    if failing_identifier == 1:
        assert target.interval_results[0].status == "scored"
    else:
        assert target.status == "failed"
        assert target.reason == "separated_vad_error"
    assert record.channels["R"].metrics["dnsmos_p808"].interval_results[0].status == "scored"


def test_dnsmos_companion_digest_controls_result_reuse(tmp_path, core_runtime) -> None:
    """A changed companion model invalidates cached DNSMOS scores."""
    write_triple(tmp_path, channels=1)
    args = ["--input_dir", str(tmp_path), "--metrics", "dnsmos_p808"]
    assert main(args) == main(args) == 0
    assert len(core_runtime.dnsmos_scores) == 1
    core_runtime.dnsmos_model_digests["dnsmos/sig_bak_ovr.onnx"] = "companion-b"
    assert main(args) == 0
    assert len(core_runtime.dnsmos_scores) == 2
    assert len({record.evaluation_id for record in results_in(tmp_path)}) == 2


@pytest.mark.parametrize(("metric", "vad_identifier"), [("visqol", 1), ("dnsmos_p808", 3)])
def test_cli_checks_detected_activity_before_scoring(tmp_path, core_runtime, metric, vad_identifier) -> None:
    """Merged gaps do not count toward the minimum, and the exact boundary qualifies."""
    write_triple(tmp_path, channels=1, sample_count=64000)
    args = ["--input_dir", str(tmp_path), "--metrics", metric]
    core_runtime.vad_intervals[vad_identifier] = (Interval(0, 24000), Interval(32000, 55999))
    core_runtime.metric_parameters[metric] = {"min_active_samples": 48000}
    assert main(args) == 0
    (record,) = results_in(tmp_path)
    (candidate,) = record.channels["T"].metrics[metric].interval_results
    assert candidate.interval == Interval(0, 55999)
    assert (candidate.status, candidate.reason) == ("ineligible", "insufficient_activity")
    assert not core_runtime.scores and not core_runtime.dnsmos_scores
    assert record.configuration.parameters[metric]["min_active_samples"] == 48000

    core_runtime.vad_intervals[vad_identifier] = (Interval(0, 24000), Interval(32000, 56000))
    assert main([*args, "--overwrite"]) == 0
    (record,) = results_in(tmp_path)
    (candidate,) = record.channels["T"].metrics[metric].interval_results
    assert candidate.interval == Interval(0, 56000)
    assert candidate.status == "scored"
    scored_waveform = core_runtime.scores[0][1] if metric == "visqol" else core_runtime.dnsmos_scores[0]
    assert scored_waveform.size == 56000

    core_runtime.metric_parameters[metric]["min_active_samples"] = 48001
    assert main(args) == 0
    records = results_in(tmp_path)
    assert len({result.evaluation_id for result in records}) == 2
    stricter = next(result for result in records if result.evaluation_id != record.evaluation_id)
    (candidate,) = stricter.channels["T"].metrics[metric].interval_results
    assert (candidate.status, candidate.reason) == ("ineligible", "insufficient_activity")


@pytest.mark.parametrize(
    "selection", [[], ["--metrics", "all"], ["--metrics", "speaker_cosine"], ["--metrics", "core", "wer"]]
)
def test_unimplemented_selections_fail_before_preflight(tmp_path, core_runtime, selection, caplog) -> None:
    """All remains the full protocol and cannot silently fall back to core."""
    assert main(["--input_dir", str(tmp_path), *selection]) == 1
    assert not core_runtime.loads
    assert "not implemented" in caplog.text
    assert not list(tmp_path.iterdir())


def test_missing_backend_fails_preflight_without_writes(tmp_path, core_runtime, monkeypatch, caplog) -> None:
    """A missing requested runtime prevents partial processing of a run."""
    write_triple(tmp_path)

    def unavailable():
        """Mimic an unavailable optional dependency."""
        raise ImportError("test missing pystoi")

    monkeypatch.setattr(pipeline, "Stoi", unavailable)
    assert main(["--input_dir", str(tmp_path), "--metrics", "core"]) == 1
    assert "preflight failed" in caplog.text
    assert not list(tmp_path.glob("*_results_*.json"))
    assert not list(tmp_path.glob("summary_*.json"))


def test_empty_selection_and_permutation_rejection(tmp_path, core_runtime) -> None:
    """An empty selected corpus succeeds; permutation remains a configuration error."""
    assert main(["--input_dir", str(tmp_path), "--metrics", "activity"]) == 0
    assert core_runtime.loads == {"vad": 1}
    assert not list(tmp_path.iterdir())
    with pytest.raises(SystemExit) as exc:
        main(["--input_dir", str(tmp_path), "--compute_permutation"])
    assert exc.value.code == 2


def test_short_recording_preserves_ineligible_overlap_candidate(tmp_path, core_runtime) -> None:
    """Recordings under 0.5 s retain overlap coverage without calling SI-SDR."""
    for name, values in (
        ("short.wav", np.linspace(5, 5.2, 4800)),
        ("short_reference.wav", np.stack((np.linspace(1, 1.2, 4800), np.linspace(2, 2.2, 4800)), axis=1)),
        ("short_separated_hash.wav", np.linspace(3, 3.2, 4800)),
    ):
        sf.write(tmp_path / name, values, 16000, subtype="FLOAT")
    assert main(["--input_dir", str(tmp_path), "--metrics", "overlap_si_sdri"]) == 0
    (record,) = results_in(tmp_path)
    (candidate,) = record.channels["T"].metrics["overlap_si_sdri"].interval_results
    assert candidate.status == "ineligible"
    assert candidate.reason == "overlap_span_too_short"
    assert not core_runtime.scores


@pytest.mark.parametrize("active_samples", [3199, 3200])
def test_sisdr_overlap_activity_boundary(tmp_path, core_runtime, active_samples) -> None:
    """Keep the ordinary utterance minimum and score overlap from 0.2 seconds."""
    write_triple(tmp_path)
    core_runtime.vad_intervals[2] = (Interval(0, active_samples),)
    assert main(["--input_dir", str(tmp_path), "--metrics", "si_sdri", "overlap_si_sdri"]) == 0
    (record,) = results_in(tmp_path)
    residual = record.channels["R"]
    assert not residual.metrics["si_sdri"].interval_results
    overlap = residual.metrics["overlap_si_sdri"]
    if active_samples == 3200:
        (score,) = overlap.interval_results
        assert score.interval == Interval(0, 8000)
        assert score.status == "scored"
    else:
        assert not overlap.interval_results


def test_silent_outputs_keep_missing_utterance_diagnostics(tmp_path, core_runtime) -> None:
    """Missing speech is measured on reference intervals independently of output VAD."""
    write_triple(tmp_path)
    sf.write(tmp_path / "call_name_separated_config_with_suffix.wav", np.zeros(16000), 16000, subtype="FLOAT")
    assert main(["--input_dir", str(tmp_path), "--metrics", "activity"]) == 0
    (record,) = results_in(tmp_path)
    target = record.channels["T"]
    assert target.separated_vad == ()
    assert target.separated_missing_utterances == (Interval(0, 10000),)
    assert target.activity.counts.fn == 6
    assert target.metrics["si_sdri"].status == "disabled"


def test_no_speech_and_all_invalid_reports(tmp_path, core_runtime) -> None:
    """Known inference IDs retain zero-count reports for empty candidate populations."""
    write_triple(tmp_path)
    for path in tmp_path.glob("*.wav"):
        samples = np.zeros((16000, 2)) if "reference" in path.name else np.zeros(16000)
        sf.write(path, samples, 16000, subtype="FLOAT")
    args = ["--input_dir", str(tmp_path), "--metrics", "core"]
    assert main(args) == 0
    (summary_file,) = tmp_path.glob("summary_*.json")
    summary = json.loads(summary_file.read_text())
    assert summary["channels"]["T"]["metrics"]["si_sdri"]["distribution"]["mean"] is None
    assert summary["channels"]["T"]["activity"]["precision"] is None
    assert not core_runtime.scores
    (record,) = results_in(tmp_path)
    assert record.reference_overlap == ()
    (tmp_path / "call_name_reference.wav").unlink()
    assert main(args) == 1
    summary = json.loads(summary_file.read_text())
    assert summary["recordings"]["invalid"] == 1
    assert summary["recordings"]["processed"] == 0


def test_ambiguous_outputs_and_missing_directory_fail(tmp_path, core_runtime, caplog) -> None:
    """Do not overwrite a result when two audio formats have the same inference suffix."""
    write_triple(tmp_path)
    values = np.zeros(16000, dtype=np.float32)
    sf.write(tmp_path / "call_name_separated_config_with_suffix.flac", values, 16000)
    assert main(["--input_dir", str(tmp_path), "--metrics", "core"]) == 1
    assert "share an inference suffix" in caplog.text
    assert not list(tmp_path.glob("*_results_*.json"))
    assert main(["--input_dir", str(tmp_path / "missing"), "--metrics", "activity"]) == 1
    (tmp_path / "call_name_separated_config_with_suffix.flac").unlink()
    sf.write(tmp_path / "call_name.flac", values, 16000)
    assert main(["--input_dir", str(tmp_path), "--metrics", "activity"]) == 1
    assert "share a stem" in caplog.text


def test_vad_failure_reason_survives_disabled_activity(tmp_path, core_runtime) -> None:
    """A failed VAD remains diagnosable in interval-only selections and cannot be reused."""
    write_triple(tmp_path)
    core_runtime.vad_failure = 3
    args = ["--input_dir", str(tmp_path), "--metrics", "stoi"]
    assert main(args) == 1
    (record,) = results_in(tmp_path)
    assert record.channels["T"].activity.status == "disabled"
    assert record.channels["T"].separated_vad_error == "test VAD failure"
    assert record.channels["T"].metrics["stoi"].interval_results[0].status == "scored"
    core_runtime.vad_failure = None
    assert main(args) == 0


@pytest.mark.parametrize("failed_identifier", [1, 2])
def test_reference_vad_failure_preserves_independent_populations(tmp_path, core_runtime, failed_identifier) -> None:
    """Failed reference VAD never blocks output VAD or the other reference channel."""
    write_triple(tmp_path, variants=("one", "two"))
    core_runtime.vad_failure = failed_identifier
    assert main(["--input_dir", str(tmp_path), "--metrics", "core"]) == 1
    assert core_runtime.vad_calls == [1, 2, 3, 4, 3, 4]
    failed_channel = "T" if failed_identifier == 1 else "R"
    healthy_channel = "R" if failed_channel == "T" else "T"
    for record in results_in(tmp_path):
        failed = record.channels[failed_channel]
        healthy = record.channels[healthy_channel]
        assert failed.reference_vad_error == "test VAD failure"
        assert failed.reference_active is None
        assert failed.separated_vad_error is None
        assert failed.separated_vad and failed.separated_utterances
        assert failed.reference_utterances == ()
        assert failed.activity.status == "failed"
        assert healthy.activity.status == "scored"
        assert healthy.metrics["stoi"].interval_results[0].status == "scored"
        assert record.reference_overlap is None
        assert record.status == "failed"


def test_metric_selection_resolves_once(tmp_path, core_runtime, monkeypatch) -> None:
    """Pipeline construction owns selection expansion and preflight validation."""
    from unittest.mock import Mock

    resolve = Mock(wraps=pipeline._resolve_metrics)
    monkeypatch.setattr(pipeline, "_resolve_metrics", resolve)
    assert main(["--input_dir", str(tmp_path), "--metrics", "core"]) == 0
    resolve.assert_called_once_with(["core"])


def test_both_vad_errors_survive_disabled_activity(tmp_path, core_runtime) -> None:
    """Reference and output failures remain distinguishable in the same channel."""
    write_triple(tmp_path)
    core_runtime.vad_failure = (1, 3)
    assert main(["--input_dir", str(tmp_path), "--metrics", "stoi"]) == 1
    (record,) = results_in(tmp_path)
    target = record.channels["T"]
    assert target.reference_vad_error == target.separated_vad_error == "test VAD failure"
    assert target.activity.status == "disabled"
    assert core_runtime.vad_calls == [1, 2, 3, 4]
    assert record.channels["R"].separated_utterances
    assert record.channels["R"].metrics["stoi"].interval_results[0].status == "scored"


@pytest.mark.parametrize(
    ("failures", "utterance_channels", "overlap_available", "activity_channels"),
    [
        ((2,), "T", False, "T"),
        ((1,), "R", False, "R"),
        ((1, 2), "", False, ""),
        ((4,), "TR", True, "T"),
        ((3,), "TR", True, "R"),
        ((3, 4), "TR", True, ""),
        ((2, 4), "T", False, "T"),
        ((1, 3), "R", False, "R"),
        ((1, 2, 3, 4), "", False, ""),
    ],
)
def test_summary_measurement_exclusions_for_vad_failures(
    tmp_path, core_runtime, failures, utterance_channels, overlap_available, activity_channels
) -> None:
    """All nine failure combinations preserve independent participation and exclusions."""
    write_triple(tmp_path)
    core_runtime.vad_failure = failures
    assert main(["--input_dir", str(tmp_path), "--metrics", "core"]) == 1
    (summary_path,) = tmp_path.glob("summary_*.json")
    summary = json.loads(summary_path.read_text())
    (record,) = results_in(tmp_path)
    for channel in ("T", "R"):
        measurements = summary["channels"][channel]
        for metric in ("si_sdri", "stoi", "stoii", "overlap_si_sdri"):
            participates = overlap_available if metric == "overlap_si_sdri" else channel in utterance_channels
            coverage = measurements["metrics"][metric]["recording_coverage"]
            assert coverage["candidate_recordings"] == 1
            assert coverage["scored_recordings"] == int(participates)
            assert coverage["excluded_due_to_errors"] == int(not participates)
            assert coverage["no_candidate_recordings"] == 0
            if not participates:
                reason = "reference_overlap_vad_error" if metric == "overlap_si_sdri" else "reference_vad_error"
                assert coverage["exclusion_reasons"] == {reason: 1}
                assert record.channels[channel].metrics[metric].status == "failed"
        activity = measurements["activity"]["recording_coverage"]
        assert activity["scored_recordings"] == int(channel in activity_channels)
        assert activity["excluded_due_to_errors"] == int(channel not in activity_channels)
    assert core_runtime.vad_calls == [1, 2, 3, 4]
