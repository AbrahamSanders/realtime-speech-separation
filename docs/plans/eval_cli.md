# Evaluation CLI implementation plan

Status: planned; runtime implementation has not started.

## Objective and scope

Implement the [evaluation protocol](../evaluation-protocol.md) through
[`rtss.eval`](../../src/realtime_speech_separation/eval_cli.py). Keep metric
definitions and motivations in the protocol, implementation details here, and
CLI usage in the README.

Evaluation uses the supplied mixture, reference, and separated WAVs plus local
models and runtime libraries. It must not depend on dataset VAD, source recordings,
metadata, enrollment audio, MagiCodec, or separator resources.

Preserve recursive discovery, input filtering, multiple inference configurations,
resumability, per-separation JSON, and JSON/CSV summaries. Support target-only
separations using the protocol's channel conventions. Full-set and source-dataset
reports must coexist; see [Reporting](../evaluation-protocol.md#reporting) for why
dataset-specific results are required.

## Changes from the current CLI

The main files are [`eval_cli.py`](../../src/realtime_speech_separation/eval_cli.py),
[`data_utils.py`](../../src/realtime_speech_separation/utils/data_utils.py),
[`pyproject.toml`](../../pyproject.toml), and
[`test_package.py`](../../tests/test_package.py).

- Remove unused enrollment loading and replace full-recording Asteroid scoring
  with protocol preprocessing and metric adapters.
- Restrict separated-file discovery to audio files and parse the entire inference
  suffix relative to the mixture stem.
- Replace flat channel vectors with interval observations, raw counts, and
  metric-specific coverage. Pool observations instead of recording means, and
  preserve other scores when one metric fails.
- Replace unconditional reuse of existing JSON with configuration/source checks.
  Keep filtered summaries limited to the selected inputs, including cached ones.
- Handle empty selections and unavailable metrics without first-result indexing
  or nonfinite JSON values.

## Libraries and resources

Resolve compatible versions in each owning stage, using local Python 3.12 and
checking the package's supported Python versions. Keep runtime dependencies out
of base/dev installs.

| Component | Implementation | Integration notes |
| --- | --- | --- |
| Audio I/O | Existing librosa, NumPy, and soundfile. | Reuse float32 loading and resampling. |
| VAD | `silero_vad.load_silero_vad` and `get_speech_timestamps`. | Use the packaged model, waveform input, and protocol settings. [API](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py). |
| SI-SDR | TorchMetrics functional `scale_invariant_signal_distortion_ratio`. | Retain individual scores and compute improvement from paired calls. [API](https://lightning.ai/docs/torchmetrics/stable/audio/scale_invariant_signal_distortion_ratio.html). |
| STOI | `pystoi.stoi(reference, estimate, 16000, extended=False)`. | Distinguish insufficient-speech warnings from valid zero scores. [Implementation](https://github.com/mpariente/pystoi/blob/master/pystoi/stoi.py). |
| PESQ | `pesq.pesq(16000, reference, estimate, mode="wb")`, exception mode. | Native extension; check Python/NumPy and build compatibility. [Wrapper](https://github.com/ludlows/PESQ). |
| ViSQOL | Official Python API, speech mode and matching model asset. | Native installation; use the array API. [Setup/API](https://github.com/google/visqol#python-api-usage). |
| DNSMOS-P.808 | TorchMetrics `deep_noise_suppression_mean_opinion_score`, `personalized=False`. | Select P.808 from the four returned components; add CPU ONNX Runtime and requests. [API](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/dnsmos.py). |
| Speaker embeddings | WeSpeaker multilingual VoxBlink2 SimAMResNet34 ONNX model. | Use CPU ONNX Runtime and the official `torchaudio.compliance.kaldi` frontend; pin matching torch/torchaudio builds. [Models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md?plain=1), [frontend](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/infer_onnx.py). |
| Transcription | In-process `pywhispercpp.model.Model`, local `ggml-large-v3-turbo.bin`. | Pass 16 kHz float32 arrays; check binding/native build compatibility. [API](https://github.com/absadiki/pywhispercpp/blob/main/pywhispercpp/model.py). |
| WER | `jiwer.process_words`, JiWER 4+. | Retain edit counts, including insertions against empty references. [Usage](https://jitsi.github.io/jiwer/usage/). |
| Intervals, activity, cosine, aggregation | Small NumPy/Python helpers. | Reuse libraries for signal metrics, features, neural inference, and edit distance. |

Use `eval` for Silero/TorchMetrics/pystoi, `eval-quality` for PESQ/DNSMOS,
`eval-speaker` for the speaker frontend, and `eval-asr` for pywhispercpp/JiWER.
The latter extras include `eval`. Document ViSQOL native setup separately.
Replace Asteroid when core metrics land; retain pandas only if reporting uses it.
Avoid broad toolkit extras and changes to unrelated dependency groups.

## Remaining protocol decisions

Settle these in Stage 1 and record accepted metric semantics in the protocol.
Existing protocol rules, including silence handling and prototype scope, remain
authoritative.

- **Overlap padding:** at recording boundaries, move unavailable padding to the
  other side. If the recording cannot provide a 0.5-second span, mark the region
  ineligible.
- **Activity frames:** a strict majority means more than 50%; ties are inactive.
  Use the actual duration of the final partial frame.
- **SI-SDR:** use `zero_mean=True`; record dtype and backend epsilon behavior.
- **Prototypes:** average raw reference embeddings with equal utterance weight,
  then apply ordinary cosine normalization. A failed required reference embedding
  makes that channel's prototype unavailable.
- **WER:** apply the same Unicode NFC, lowercase, punctuation removal, whitespace
  normalization, and whitespace tokenization to both transcripts. Detect language
  from each reference channel and use it for its separation, with an explicit
  language override available. Use transcription rather than translation.
  Corpus WER is `null` when total reference words are zero; otherwise include all
  insertions, including those from empty references. Document whitespace WER's
  language limitations and label it as agreement with reference ASR.

## Implementation structure

Keep argument parsing, logging, and discovery in `eval_cli.py`. Put the pipeline
under `src/realtime_speech_separation/evaluation/`:

- `types.py`: typed configuration, interval, observation, and result records.
- `audio.py`: triple loading and validation.
- `vad.py`: Silero adapter, interval operations, utterances, overlap, and activity.
- `metrics.py`: SI-SDR/STOI/PESQ/ViSQOL/DNSMOS adapters.
- `speaker.py` and `asr.py`: speaker and transcription adapters.
- `reporting.py`: serialization, result reuse, aggregation, and CSV output.
- `pipeline.py`: per-recording preparation, metric calls, and resource reuse.

### Shared backend and error handling

Use small functions with injectable backend calls and a fixed metric dispatch
table. Keep optional imports inside selected adapters so package import and CLI
help work without evaluation backends.

- Preflight requested packages/assets and load each model/session once. Default
  to CPU with bounded threads and inference mode. Provision assets separately;
  evaluation must run offline without automatic downloads or GPU selection.
- Validate audio once at the loading boundary. Represent missing/unreadable files
  and invalid waveforms with one recording-error record containing path and
  reason; reuse decoder exceptions for file errors.
- Map documented scoring exceptions/warnings to observation statuses and reasons.
  Convert nonfinite scores to failed observations with `null` values. Preserve
  other metrics and channels. Do not hide unexpected programming errors.
- Use strict JSON (`allow_nan=False`) and atomic per-result writes. Return nonzero
  for configuration, invalid-input, or requested-backend execution failures;
  valid ineligibility and mathematically undefined scores are reportable outcomes.
- Reuse reference VAD, baselines, embeddings, and transcripts across inference
  variants in memory. Keep this reuse local to the mixture being evaluated.
  Persistent intermediate caches are deferred.

### Results and aggregation

Define one schema containing:

- Evaluator revision, inference hash, evaluation ID, resolved configuration,
  backend versions, and model content digests.
- Relative input paths, source content hashes, native rates/channel counts,
  evaluation rate/sample count, and available output channels.
- Raw VAD, utterance, and reference-overlap intervals with IDs, sample bounds,
  and original active duration. Derive spans from bounds.
- Activity TP/FP/FN and frame counts; WER S/D/I/reference-word counts, status,
  and raw/normalized full-channel transcripts.
- T/R observations for `si_sdri`, `overlap_si_sdri`, `stoi`, `stoii`, `pesq`,
  `visqol`, `dnsmos_p808`, and `speaker_cosine`. Each has an interval ID/origin,
  status (`scored`, `ineligible`, `undefined`, or `failed`), value or `null`,
  and reason where needed. Improvement records retain input/output scores.
- Reference-prototype eligibility/extraction records and silent-estimate
  diagnostics. Disabled metrics and unavailable channels have explicit statuses.

Derive coverage counts and reason totals from observations during reporting;
maintain no parallel mutable counters. Use the protocol's candidate/eligible/scored
rules. Candidates are reference utterances for intrusive metrics, reference
overlap regions for overlap SI-SDRi, and separated utterances for DNSMOS/similarity.
Report activity/WER coverage in recording, frame, and word units as appropriate.

Group summaries by inference hash and evaluation ID. Apply the protocol's pooled
aggregation independently per channel, metric, and overall/overlap condition.
Use NumPy's `linear` percentile method and record it in configuration. Compute
improvement distributions from paired observations. Derive recording counts for
processed, invalid, reused, target-only, and failed cases, along with inactive
reference channels and active output frames on those channels.

The evaluation ID hashes the evaluator revision, selected metrics, score/report
parameters, backend versions, and model digests. Record exposed native-library
versions where relevant; exhaustive native-build fingerprints are deferred.
Source audio hashes are separate result-validity keys so recordings can share
an evaluation ID. Reuse only schema-compatible results with matching IDs/hashes;
recompute stale, legacy, interrupted, or failed results. Valid ineligible or
undefined observations do not prevent reuse. `--overwrite` forces recomputation.

### Full-set and dataset reports

Use these names:

- `<file_id>_results_<inference_hash>_<evaluation_id>.json`
- `summary_<inference_hash>_<evaluation_id>_<selection_id>.json`
- `metrics_<inference_hash>_<evaluation_id>_<selection_id>.csv`

`--input_filter` constrains discovery, processing, and aggregation, including
cached results. Summarize only results reached through matching mixtures; do not
scan arbitrary result JSONs. Preserve the existing case-insensitive substring
matching and OR semantics for multiple filter terms.

Use `selection_id=all` without a filter, otherwise a digest of lowercased, sorted,
deduplicated filter terms. Include those terms and actual selected mixture paths
in report provenance. Keep selection out of the evaluation ID so per-separation
results can be reused for full validation/test reports and source-dataset reports.
These report files must coexist. Select datasets through their paths; evaluation
still requires no dataset metadata.

The CSV has one row per separation with T/R scalar summaries, raw counts,
statuses, and coverage. Corpus summaries use raw observations/counts rather than
CSV recording means. Detailed observations remain in JSON; a separate utterance
CSV is deferred.

## Implementation stages

Stages 1–4 run in order. Stages 5–9 depend on Stage 4 and can be implemented in
any order; Stage 10 finishes integration. Each stage uses the shared contracts
above. Record completion, checks, and deviations here as work proceeds.

### Stage 1 — Settle decisions and add reporting foundations

- Read the protocol, current CLI, package tests, and dependency manifest. Settle
  the remaining protocol decisions above, keeping CLI usage in the README.
- Add `types.py` and pure reporting functions: schema, evaluator revision marker,
  strict serialization, configuration identity, coverage, and pooled aggregation.
- Keep reporting tests runnable with base/dev dependencies. Resolve runtime
  dependencies in their owning stages.

**Tests:** hand-built records with unequal utterance counts, activity/WER from
summed counts, partial metric failures, missing residuals, zero denominators,
empty/singleton distributions, and JSON round trips. Test compatible-result
parsing and configuration/source mismatches here rather than per backend.

### Stage 2 — Load audio triples and fix discovery

- Read CLI discovery, `data_utils.py`, `audio_utils.py`, and inference delay
  removal in `inference_cli.py`.
- Add float32 16 kHz loading in `audio.py`. Load mixture mono, require reference
  `(2, n)`, and accept separation `(n,)`, `(1, n)`, or `(2, n)`, normalizing mono
  to `(1, n)`. Reject unsupported channel counts and empty/nonfinite waveforms.
- Require equal resampled lengths, including across different native rates.
  Alignment is a caller guarantee: do not shift, truncate, pad, normalize, or clip
  recordings. Inference has already removed its configured delay.
- Remove enrollment loading. Match adjacent separated audio by mixture stem and
  supported extension, ignore sidecars/directories, sort deterministically, and
  retain the entire inference suffix. Preserve input filtering locally to eval;
  keep any backend files outside discovery paths.

**Tests:** small temporary WAV triples covering mono/stereo estimates, one
resampling case, absent enrollment, unreadable/missing audio, invalid waveforms,
wrong channel counts, and mismatched lengths. Use one nested discovery fixture
with sidecars, multiple inference suffixes, and filters.

### Stage 3 — Add VAD, intervals, and activity

- Add the Silero adapter in `vad.py`, resetting state per channel/recording and
  passing the protocol's explicit parameters. Run it on reference T/R and each
  available separated channel.
- Represent intervals as sorted, disjoint half-open sample bounds `[start, end)`.
  Check backend intervals once, then use small merge/intersection helpers.
- Construct utterances and overlap from raw activity using protocol thresholds.
  Active duration cannot exceed span, so the active-duration threshold also
  enforces the matching span minimum. Preserve active duration across merged
  gaps, and apply overlap padding after filtering.
- Construct `ceil(n / 1600)` activity frames and TP/FP/FN for each available pair,
  using the agreed majority and final-frame rules.

**Tests:** inject fixed intervals covering exact merge/active-duration thresholds,
long spans with insufficient activity, fragmented/false overlaps, boundary
padding, no speech/overlap, tied frames, and final partial frames. A fake Silero
adapter verifies channel calls, state reset, and reference reuse.

### Stage 4 — Integrate core metrics and the CLI

- Add functional SI-SDR and STOI adapters. Slice reference/estimate/mixture at
  identical bounds, preserving channel identity. Apply the protocol's centered
  zero-energy handling and distinguish valid zero STOI from insufficient-frame
  warnings. Reuse mixture baselines and retain paired input/output scores.
- Wire loading, preprocessing, metrics, results, and reports through `pipeline.py`.
  Add the evaluation/selection IDs, source fingerprints, and result reuse rules.
- Add `--metrics all` (default), `core` (SI-SDRi, activity, STOI, STOIi), and
  individual metric choices. Reject unavailable requested metrics, including
  not-yet-implemented families. Label subsets as partial protocol evaluations.
- Preserve input/filter/overwrite/logging flags. Keep `--sample_rate=16000` for
  compatibility and reject other rates and `--compute_permutation`.
- Replace Asteroid with explicit Silero/TorchMetrics/pystoi dependencies in `eval`.

**Tests:** use known adapter returns to check argument order, paired differences,
fixed T/R assignment, interval-specific baselines, silent estimates/zero-energy
references, valid zero STOI, and unavailable baselines. Keep one real SI-SDR
sanity check that mixture-as-estimate gives zero improvement in an opt-in backend
test. Start a synthetic CLI fixture with fake VAD/metrics, then extend it through
later stages. Cover shared resource preflight and failure isolation once.

### Stage 5 — Add PESQ

- Add wideband PESQ over reference utterances and include it in `eval-quality`.
  Document native setup and resolved Python/NumPy compatibility.
- Map documented no-utterance/too-short exceptions to observation reasons using
  the selected exception mode.

**Tests:** verify PESQ arguments and exception mapping with a fake backend.
An opt-in wrapper smoke test checks a valid signal and documented error cases.

### Stage 6 — Add ViSQOL

- Initialize the official API in 16 kHz speech mode with a local speech model;
  return per-utterance MOS-LQO and document native setup.
- Establish the pinned backend's actual short-input requirements. Keep protocol
  utterances; record unsupported ones without adding padding or concatenation.
- Investigate the official CLI only if the Python API cannot run in the target
  environment; record that decision before adding a fallback adapter.

**Tests:** verify speech-mode/model configuration with a fake backend. Use an
opt-in smoke test to establish native short-utterance behavior.

### Stage 7 — Add DNSMOS-P.808

- Add the TorchMetrics adapter and CPU ONNX Runtime/requests to `eval-quality`.
  Select only `p808_mos`, preserving the library's short-input extension and
  internal-window averaging rather than adding an utterance-duration cutoff.
- Document and checksum all assets needed by the selected API, including any
  companion models. Provision its `~/.torchmetrics/DNSMOS` cache for offline use;
  the current API has no public model-path argument. Reconsider the adapter only
  if that location cannot meet deployment needs.

**Tests:** a fake four-component return verifies P.808 selection and separated
utterance bounds. An opt-in backend smoke test covers short and long utterances
with pre-provisioned assets.

### Stage 8 — Add WeSpeaker similarity

- Add `speaker.py` for the selected SimAMResNet34 ONNX model and local model path.
  Reuse the official filterbank frontend, amplitude scaling, frame settings, and
  mean normalization; add matching torchaudio/CPU ONNX Runtime to `eval-speaker`.
- Implement the protocol's reference-prototype and separated-utterance rules,
  including the agreed embedding average. Report unavailable prototypes,
  zero-norm embeddings, and extraction failures.
- Return raw cosine in `[-1, 1]`, not WeSpeaker's convenience `[0, 1]` transform.
  Preserve upstream attribution when adapting its inference example.

**Tests:** deterministic fake embeddings verify per-recording/channel scope,
equal weighting, active-duration versus span eligibility at the 3-second boundary,
missing prototypes, and negative cosine. Before accepting this frontend, compare
embeddings with the official extractor using the same model/configuration in an
opt-in local check.

### Stage 9 — Add transcription and WER

- Add `asr.py` using in-process pywhispercpp and a local large-v3-turbo model;
  add pywhispercpp/JiWER 4+ to `eval-asr`. Record decoding settings and quantization.
- Transcribe full channels as specified by the protocol. Pass contiguous mono
  float32 arrays directly to `Model.transcribe` and concatenate returned segments.
- Apply the agreed language policy and deterministic settings where available.
  Reset decode parameters and context/prompts between independent channels/files;
  verify the pinned binding's language-detection API.
- Normalize transcripts and retain S/D/I and reference words
  (`hits + substitutions + deletions`), with raw/normalized text and
  recognizer/normalizer identities.
  Rely on documented binding returns/errors; timeouts and cancellation machinery
  are outside this implementation.

**Tests:** fake model outputs and known text pairs cover edit counts, empty
references/hypotheses, Unicode/punctuation normalization, language consistency,
and no prompt/parameter leakage. Keep real large-v3-turbo transcription opt-in.

### Stage 10 — Finish reports, documentation, and integration

- Complete full-protocol selection and the shared result/reuse/report contracts.
  Handle no discovered mixtures/separations and all-invalid/all-ineligible runs
  with useful diagnostics and zero-count reports where an inference hash is known.
- Finish the recording CSV and full-set/dataset reports. Update the README,
  architecture evaluation links (including `evaluation_protocol.md`), and output
  consumers such as shell scripts and run-eval documentation. Document native
  setup/assets, offline operation, subsets, naming/schema, coverage, and costs.

**Tests:** extend the Stage 4 CLI fixture to all metric families. Verify multiple
nested mixtures/inference hashes, core/full selection, target-only output,
empty distributions, exit statuses, and valid JSON/CSV. Check cached reruns,
changed audio/configuration/model identities, and overwrite using that fixture.
Run dataset-filtered summaries after a full-set run, using two-speaker and
multiparty dataset paths with distinct known scores. Cached results must respect
each selection and all reports must coexist. Check the full-set summary remains
unchanged by subsequent dataset reports. Keep aggregation boundary cases
in Stage 1 tests rather than duplicating them throughout integration tests.

## Validation during implementation

Add focused tests as each stage lands and run the repository checks described in
[AGENTS.md](../../AGENTS.md#code-and-checks), using its development environment.
Keep ordinary tests offline and CPU-only with injected backends and temporary
synthetic audio; real model/native-backend tests are opt-in. Record checks that
could not run, and keep runtime stacks out of base/dev CI.

For the final real-data smoke test, copy a few existing triples to a temporary
location and use `--input_filter` and `--overwrite` there. Verify evaluation works
without the project's `data` tree or enrollment files. Follow repository GPU
etiquette for any explicitly requested GPU check; evaluation testing does not
require separator inference or source-data changes.
