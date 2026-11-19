# Evaluation CLI implementation plan

Stages 1–8 are implemented. The ASR backend remains planned.

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
- Replace flat channel vectors with interval results, raw counts, and
  metric-specific coverage. Pool interval scores instead of recording means, and
  preserve other scores when one metric fails.
- Replace unconditional reuse of existing JSON with configuration/source checks.
  Keep filtered summaries limited to the selected inputs, including cached ones.
- Handle empty selections and unavailable metrics without first-result indexing
  or nonfinite JSON values.

## Libraries and resources

Resolve compatible versions in each owning stage, using local Python 3.12 and
checking the package's supported Python versions. Keep runtime dependencies out
of base/dev installs.

A backend is the external implementation called by an evaluation adapter:
metric backends compute scores (for example, TorchMetrics or PESQ), the VAD
backend detects speech (Silero), the speaker backend extracts embeddings
(WeSpeaker with ONNX Runtime), and the ASR backend transcribes audio
(pywhispercpp). Qualify backend references by the component they implement.

| Component | Implementation | Integration notes |
| --- | --- | --- |
| Audio I/O | Existing librosa, NumPy, and soundfile. | Reuse float32 loading and resampling. |
| VAD | `silero_vad.load_silero_vad` and `get_speech_timestamps`. | Use the packaged model, waveform input, and protocol settings. [API](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py). |
| SI-SDR | TorchMetrics functional `scale_invariant_signal_distortion_ratio`. | Retain individual scores and compute improvement from paired calls. [API](https://lightning.ai/docs/torchmetrics/stable/audio/scale_invariant_signal_distortion_ratio.html). |
| STOI | `pystoi.stoi(reference, estimate, 16000, extended=False)`. | Distinguish insufficient-speech warnings from valid zero scores. [Implementation](https://github.com/mpariente/pystoi/blob/master/pystoi/stoi.py). |
| PESQ | `pesq.pesq(16000, reference, estimate, mode="wb")`, exception mode. | Native extension; check Python/NumPy and build compatibility. [Wrapper](https://github.com/ludlows/PESQ). |
| ViSQOL | Official Python API, speech mode and matching model asset. | Native installation; use the array API. [Setup/API](https://github.com/google/visqol#python-api-usage). |
| DNSMOS-P.808 | TorchMetrics `deep_noise_suppression_mean_opinion_score`, `personalized=False`. | Select P.808 from the four returned components; add CPU ONNX Runtime and requests. [API](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/dnsmos.py). |
| Speaker embeddings | WeSpeaker multilingual VoxBlink2 SimAMResNet34 ONNX model. | Use CPU ONNX Runtime and the official `torchaudio.compliance.kaldi` frontend with ABI-compatible PyTorch and TorchAudio builds. [Models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md?plain=1), [frontend](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/infer_onnx.py). |
| Transcription | In-process `pywhispercpp.model.Model`, local `ggml-large-v3-turbo.bin`. | Pass 16 kHz float32 arrays; check binding/native build compatibility. [API](https://github.com/absadiki/pywhispercpp/blob/main/pywhispercpp/model.py). |
| WER | `jiwer.process_words`, JiWER 4+. | Retain edit counts, including insertions against empty references. [Usage](https://jitsi.github.io/jiwer/usage/). |
| Intervals, activity, cosine, aggregation | Small NumPy/Python helpers. | Reuse libraries for signal metrics, features, neural inference, and edit distance. |

Use `eval` for Silero, TorchMetrics, pystoi, PESQ, DNSMOS, and the speaker
frontend. The planned `eval-asr` extra includes `eval` and adds pywhispercpp
and JiWER. Document ViSQOL native setup separately.
Replace Asteroid when core metrics land; retain pandas only if reporting uses it.
Avoid broad toolkit extras and changes to unrelated dependency groups.

## Accepted protocol decisions

Settled in Stage 1 and recorded in the [protocol](../evaluation-protocol.md).
The following is the implementation checklist; the protocol owns the definitions.

- **Overlap padding:** at recording boundaries, move unavailable padding to the
  other side. Evaluation requires at least one second of audio per file.
- **Activity frames:** a strict majority means more than 50%; ties are inactive.
  Use the actual duration of the final partial frame.
- **SI-SDR:** use `zero_mean=True`; record dtype and metric-backend epsilon behavior.
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

- `types.py`: typed configuration, interval, speech-region, and metric/count result records.
- `audio.py`: triple loading and validation.
- `vad.py`: Silero adapter, interval operations, utterances, overlap, and activity.
- `metrics.py`: SI-SDR/STOI/PESQ/ViSQOL/DNSMOS adapters.
- `speaker.py` and `asr.py`: speaker and transcription adapters.
- `reporting.py`: serialization, result reuse, aggregation, and CSV output.
- `pipeline.py`: per-recording preparation, metric calls, and resource reuse.

### Metric, VAD, speaker, and ASR backend error handling

Use small functions with injectable metric, VAD, speaker, and ASR backend calls
and a fixed metric dispatch table. Keep optional imports inside selected adapters
so package import and CLI help work without those external libraries.

- Preflight requested packages/assets and load each model/session once. Default
  to CPU and use backend-provided inference handling. Leave runtime thread
  configuration to the user and external libraries. Reuse local assets for offline
  evaluation. The default WeSpeaker export downloads during preflight when its
  verified cache entry is missing. Evaluation uses CPU.
- Validate audio once at the loading boundary. Represent missing/unreadable files
  and invalid waveforms with one recording-error record containing path and
  reason; reuse decoder exceptions for file errors.
- Map documented scoring exceptions/warnings to interval-result statuses and reasons.
  Convert nonfinite scores to failed interval results with `null` values. Preserve
  other metrics and channels. Do not hide unexpected programming errors.
- Use strict JSON (`allow_nan=False`) and atomic per-result writes. Return nonzero
  for configuration, invalid-input, or requested metric/VAD/speaker/ASR backend failures;
  valid ineligibility and mathematically undefined scores are reportable outcomes.
- Reuse reference VAD, baselines, embeddings, and transcripts across inference
  variants in memory. Keep this reuse local to the mixture being evaluated.
  Persistent intermediate caches are deferred.

### Results and aggregation

Define one schema containing:

- Package version, inference hash, evaluation ID, resolved configuration,
  metric/VAD/speaker/ASR backend versions, and model content digests.
- Relative input paths, source content hashes, native rates/channel counts,
  evaluation rate/sample count, and channel availability. Derive output channels
  from availability rather than storing a second channel list in JSON.
- Raw VAD intervals with sample bounds. Utterance and reference-overlap
  `SpeechRegion` records reference an `Interval` and retain original active
  samples separately, excluding merged gaps and added padding. Raw VAD intervals
  are fully active and need no separate active-sample count. Derive spans from
  bounds.
- Activity TP/FP/FN and frame counts; WER S/D/I/reference-word counts, status,
  and raw/normalized full-channel transcripts.
- T/R `IntervalMetricResult` records for `si_sdri`, `overlap_si_sdri`, `stoi`, `stoii`,
  `pesq`, `visqol`, `dnsmos_p808`, and `speaker_cosine`. Each references its interval,
  with `interval_source` specifying how its bounds were selected,
  status (`scored`, `ineligible`, `undefined`, or `failed`), value or `null`,
  and reason where needed. Improvement records retain input/output scores.
- Reference-prototype eligibility/extraction records and `separated_missing_utterances`
  diagnostics. These reference utterance intervals have corresponding separated
  waveforms with zero energy after subtracting their own arithmetic means; this is independent of
  separated VAD activity. Disabled metrics and unavailable channels have explicit
  statuses.

Derive coverage counts and reason totals from interval results during reporting;
maintain no parallel mutable counters. Use the protocol's candidate/eligible/scored
rules. Candidates are reference utterances for intrusive metrics, reference
overlap regions for overlap SI-SDRi, and separated utterances for DNSMOS/similarity.
Report activity/WER coverage in recording, frame, and word units as appropriate.

Group summaries by inference hash and evaluation ID. Apply the protocol's pooled
aggregation independently per channel, metric, and overall/overlap condition.
Use NumPy's fixed `linear` percentile method and record it in summary metadata.
Compute improvement distributions from paired interval results. Derive recording
counts for processed, invalid, reused, target-only, and failed cases, along with
inactive reference channels and active output frames on those channels.

The evaluation ID hashes the package version, selected metrics, score/report
parameters, metric/VAD/speaker/ASR backend versions, and model digests. Record
exposed native-library versions where relevant; exhaustive native-build
fingerprints are deferred.
Source audio hashes are separate result-validity keys so recordings can share
an evaluation ID. Reuse only structurally compatible results with matching IDs/hashes;
recompute stale, legacy, interrupted, or failed results. Valid ineligible or
undefined interval results do not prevent reuse. `--overwrite` forces recomputation.

Package versions identify published result formats and evaluation behavior; no
independent schema or evaluator revision counters are maintained. Developers
changing records or scoring within the same package version must clean their
local evaluation outputs and regenerate them. Track `cached_result_paths` only
for the current CLI invocation to count separations whose result JSONs were
loaded from disk, without modifying their persisted results.

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
statuses, and coverage. Corpus summaries use raw interval scores/counts rather than
CSV recording means. Detailed interval results remain in JSON; a separate utterance
CSV is deferred.

## Implementation stages

Stages 1–4 run in order. Stages 5–9 depend on Stage 4 and can be implemented in
any order; Stage 10 finishes integration. Each stage uses the shared contracts
above. Record completion, checks, and deviations here as work proceeds.

### Stage 1 — Settle decisions and add reporting foundations

- Read the protocol, current CLI, package tests, and dependency manifest. Settle
  the remaining protocol decisions above, keeping CLI usage in the README.
- Add `types.py` and pure reporting functions: result records, package-version provenance,
  strict serialization, configuration identity, coverage, and pooled aggregation.
- Keep reporting tests runnable with base/dev dependencies. Resolve runtime
  dependencies in their owning stages.

**Tests:** hand-built records with unequal utterance counts, activity/WER from
summed counts, partial metric failures, missing residuals, zero denominators,
empty/singleton distributions, and JSON round trips. Test compatible-result
parsing and configuration/source mismatches here rather than per metric, VAD,
speaker, or ASR backend.

**Implementation notes:**

- Added the `evaluation` package, which imports no metric, VAD, speaker, or ASR
  backends, with
  [`types.py`](../../src/realtime_speech_separation/evaluation/types.py) and
  [`reporting.py`](../../src/realtime_speech_separation/evaluation/reporting.py).
  Package version from `realtime_speech_separation.__version__` is recorded in
  configuration for per-separation JSON and summaries and participates in evaluation
  identity. Separate schema/evaluator counters were removed after review. Published
  versions control compatibility; developers regenerate local results after
  changing records within the same version.
- `Interval(start, end)` has no string ID or active-sample count. Raw VAD uses
  these bounds directly. `SpeechRegion(interval, active_samples)` adds original
  activity only for utterances and overlap regions. `IntervalMetricResult`,
  `ReferenceEmbeddingExtractionResult`, and `separated_missing_utterances` refer
  directly to intervals in Python; JSON embeds their bounds. `output_channels`
  is a read-only property derived from channel availability and omitted from JSON.
- `IntervalResultStatus` describes individual interval scores/extractions;
  `CountResultStatus` describes full-channel activity/WER outcomes. Docstrings
  give examples for these statuses, prototype availability, and recording outcomes.
  `interval_source` states which VAD population selected the bounds. Paired-score
  examples explain `value = output_score - input_score` for improvement metrics.
  `ReferencePrototypeExtractionResult` groups the reference embedding extraction
  outcomes. Speaker vectors remain in adapter memory; reporting records retain eligibility
  and extraction outcomes without vectors.
- Assumptions accepted in the protocol: exact merge/duration thresholds are
  inclusive; odd overlap padding puts the extra sample on the right; activity
  ties are inactive, with actual final-frame duration; SI-SDR uses float32,
  subtraction of the interval waveform's mean, and TorchMetrics' metric-backend
  dtype epsilon. Prototypes average raw embeddings with equal utterance weight
  before cosine normalization. WER deletes Unicode
  `P*` punctuation without inserting spaces and retains symbols, following NFC
  and lowercase. Reference-channel language detection is shared with its output,
  with an override; ASR performs transcription. Whitespace WER is labeled as
  agreement with reference ASR and its language limitations are documented.
- Reporting choices: zero denominators produce `null`; undefined/failed
  candidates remain eligible; disabled/unavailable metrics add no candidates.
  Percentiles use the fixed NumPy `linear` method, recorded in summary metadata;
  there is no percentile constructor parameter. Evaluation remains at 16 kHz.
  SD uses `ddof=0`, and paired improvement scores are checked against their
  retained input/output scores. Absolute components
  remain reportable when a baseline or improvement is unavailable. Prototype
  coverage is independent of separated-utterance coverage. Activity coverage
  includes candidate/scored recordings and frames; WER coverage includes scored
  reference words and empty-reference recordings. Unknown word totals after
  failed ASR are not estimated.
- Identity uses full SHA-256 over canonical JSON, including package version,
  sorted/deduplicated selected metrics, resolved parameters, fixed percentile method,
  and logical model content digests. Metric, VAD, speaker, and ASR library versions
  are recorded in `metric_backend_versions` and the singular
  `vad_backend_version`, `speaker_backend_version`, and `asr_backend_version`
  strings (null when unused). The caller must
  supply resolved settings/defaults when each adapter lands. Source paths/content
  hashes are separate reuse keys; native metadata is informational. Selection terms/paths stay out
  of evaluation identity. Filter identities lowercase/sort/deduplicate terms
  without stripping them, preserving substring semantics.
- Parsing requires all record fields and rejects unknown fields, invalid types,
  duplicate keys, and inconsistent identities. Reuse rejects results from other
  package versions, missing source hashes, and any recording or metric/VAD/speaker/ASR
  backend failure, while valid ineligible/undefined outcomes remain reusable.
  Writes use strict UTF-8 JSON and atomic replacement. Nonfinite metric-backend
  scores become failed interval results; nonfinite configuration/serialization
  values are rejected.
- `processed` counts valid evaluated recording results, including cached and
  partially failed results; `failed` counts any such result with metric/VAD/speaker/ASR
  backend failures. `invalid` is separate. `reused` counts results loaded from disk
  during this CLI invocation using `cached_result_paths`; cached JSONs are not modified.
  `separated_missing_utterances` identifies reference intervals whose separated
  waveforms have zero energy after mean subtraction, independently of VAD activity.
  Summaries require one inference/evaluation identity and reject duplicate
  separations. Selection/discovery enforcement, filenames, CSV output, and exit
  statuses remain in their planned integration stages.
- No runtime dependencies or CLI usage changed. Installation tests now import
  the reporting package in isolation and verify that optional metric, VAD,
  speaker, and ASR backends stay unloaded. Stage-specific offline tests cover
  the required aggregation/reuse
  boundaries, strict JSON, and atomic-write failure behavior.

**Checks:** in the `audio` environment on Python 3.12.13,
`python -m ruff check .`, `python -m ruff format --check .`, and
`python -m pytest` all passed after the review changes (64 tests, including 61
reporting cases). Tests cover direct interval references, separate active-sample
counts, package-version cache checks, and omission of the derived output-channel
property from JSON, alongside the original aggregation boundaries.
Pytest emitted one installed AnyIO plugin deprecation warning about `CallSpec2`.
`git diff --check` passed. No model, native metric-backend, or GPU checks are
needed for this stage; no checks were blocked.

**Deviations:** review replaced independent schema/evaluator version markers with
package-version provenance and direct interval references. Metric/VAD/speaker/ASR
backend version resolution remains in owning stages.

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
  keep metric/VAD/speaker/ASR backend assets outside discovery paths.

**Tests:** small temporary WAV triples covering mono/stereo estimates, one
resampling case, absent enrollment, unreadable/missing audio, invalid waveforms,
wrong channel counts, and mismatched lengths. Use one nested discovery fixture
with sidecars, multiple inference suffixes, and filters.

**Implementation notes:**

- Added [`audio.py`](../../src/realtime_speech_separation/evaluation/audio.py).
  `load_audio_triple` returns an `AudioTriple` with float32 samples at 16 kHz and
  native sample-rate/channel metadata, or the first `RecordingError` with the
  offending path and reason. `soundfile.info` retains native metadata and checks
  channel counts before downmixing. `librosa.load` performs float32 decoding,
  downmixing, and resampling as shown in the protocol. File errors preserve the
  decoder's message; librosa waveform-validation errors become recording errors.
  Unexpected programming errors propagate. Finite output and exact resampled
  lengths are checked at the loading boundary. Silence is valid; floating-point
  WAV storage is not enforced. No delay removal, length correction, amplitude normalization, or
  clipping is applied. Inference already removes delay and final-chunk padding.
- Assumption: preserve the previous `mono=True` behavior by accepting mono or
  stereo mixtures and averaging stereo channels. Reject more than two channels
  for all inputs; references require exactly two. Separations are always `(1, n)`
  or `(2, n)`, with the first channel assigned to T. Resampling uses librosa's
  default length convention; equality is tested after resampling, with no extra
  correction when native rates differ.
- Discovery remains in [`eval_cli.py`](../../src/realtime_speech_separation/eval_cli.py).
  A local mixture wrapper delegates include and exclude filtering to recursive
  `get_audio_files`, retaining the original exclusions (`_enrollment.`,
  `_separated_`, `_reference.`) across full paths. Filtering retains
  case-insensitive full-path substring matching, OR semantics, and unstripped
  terms. Adjacent separation discovery requires a regular audio file with the exact mixture-stem prefix
  and a nonempty inference suffix; results are sorted. The whole suffix,
  including underscores, is retained. Sidecars, directories, and nonadjacent
  outputs are ignored. Shared discovery and inference utilities are unchanged.
- Assumptions: use the existing supported extensions (`.wav`, `.flac`, `.mp3`,
  `.opus`), matching extensions without case sensitivity. Stem matching remains
  case-sensitive. The reference keeps the mixture's extension and sits beside
  it. Metric, VAD, speaker, and ASR backend assets belong outside the input tree;
  discovery does not inspect those assets or dataset metadata.
- The current CLI now uses the triple loader without enrollment loading. It
  passes only reference T for target-only output to the legacy scoring backend.
  Invalid triples are logged and skipped, valid triples continue, and invalid
  input makes the invocation return nonzero. Optional Asteroid/pandas imports
  were moved into execution so importing discovery and showing CLI help work
  with base/dev dependencies. Native metadata is available to the future pipeline;
  persistent source fingerprints, structured invalid-result JSON, configuration
  checks on reused results, and protocol metrics remain in their owning stages.
- Tests use temporary synthetic audio only. They cover shapes/channel order,
  unchanged samples and amplitudes at 16 kHz, differing native rates, PCM input,
  missing/unreadable files on each side, empty/nonfinite waveforms, unsupported
  channel counts, silence, and unequal resampled lengths. One nested discovery
  fixture covers supported extensions, multiple full inference suffixes,
  sidecars/directories, and filters. CLI tests use fake legacy scoring/reporting
  dependencies to verify target-only/stereo loading without enrollment and
  continuation after invalid audio. The isolated package-import test includes
  audio loading and discovery and checks that optional backends stay unloaded.

**Checks:** in the `audio` environment on Python 3.12.13,
`python -m ruff check .`, `python -m ruff format --check .`,
`python -m pytest` (117 tests), and `git diff --check` passed after review.
Pytest emitted the existing installed AnyIO plugin deprecation warning about
`CallSpec2` and three audioread standard-library deprecation warnings surfaced by
`librosa.load` on Python 3.12. No native metric/model or GPU checks are needed for
this stage; no checks were blocked. No dependencies changed.

**Deviations:** reject `--sample_rate` values other than 16000 now, moving that
small validation forward from Stage 4 so the current CLI cannot pass an incorrect
sample rate to scoring after the fixed-rate loader runs. Full pipeline/reporting
integration remains in Stage 4 and Stage 10.

### Stage 3 — Add VAD, intervals, and activity

- Add the Silero adapter in `vad.py`, relying on Silero to reset state per
  channel/recording and passing the protocol's explicit parameters. Run it on
  reference T/R and each available separated channel.
- Represent intervals as sorted, disjoint half-open sample bounds `[start, end)`.
  Trust Silero timestamp bounds and ordering, then use small merge/intersection
  helpers.
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

**Implementation notes:**

- Added [`vad.py`](../../src/realtime_speech_separation/evaluation/vad.py).
  `SileroVad` lazily imports its runtime and loads the packaged JIT model once
  without downloads. Calls use CPU and explicit protocol timestamp parameters.
  Silero's timestamp function owns recurrent-state reset and gradient disabling;
  the adapter does not duplicate either operation or configure threads. It exposes
  its Silero/PyTorch versions, packaged model SHA-256, and explicit settings for
  Stage 4 configuration identity. Its model is reused sequentially across
  channels and recordings; concurrent use is outside this interface.
- The `eval` extra requires `silero-vad>=6.2.1`, the tested minimum whose
  [release API](https://github.com/snakers4/silero-vad/blob/v6.2.1/src/silero_vad/utils_vad.py)
  supports the explicit exit threshold and packaged model loading. Its Python
  requirement is >=3.8, covering the project's supported versions. Base/dev and
  unrelated dependency groups are unchanged. The installed model was checked
  locally on Python 3.12.13 with PyTorch 2.13.0 and torchaudio 2.11.0.
- Trust Silero's sorted, disjoint, recording-bounded sample timestamps and
  convert them directly to `Interval` records. Removed the defensive timestamp
  validator after review. Runtime execution failures raise `VadBackendError`;
  unexpected programming errors propagate. Stage 4 owns conversion to recording
  outcomes and CLI exit status. Backend versions and model digests identify
  reproduction environments; exact dependency pins are unnecessary without a
  demonstrated incompatibility.
- `merge_regions`, `intersect_intervals`, `utterance_regions`, and
  `overlap_regions` implement the
  [protocol's timing rules](../evaluation-protocol.md#utterance-identification).
  Overlap starts from raw intersections, and filtering precedes padding.
  `SpeechRegion.active_samples` excludes gaps and padding. Assumption: when a
  recording cannot supply a 0.5-second overlap span, retain the unpadded region
  as a candidate. Stage 4 must mark its short span ineligible when constructing
  interval metric results, preserving reporting coverage rather than silently
  dropping the candidate. No result-schema change is needed.
- `activity_mask` accumulates raw activity by frame without allocating a
  sample-sized mask. `activity_counts` returns the shared `ActivityCounts`
  contract, including true-negative frames in the frame total. Frame masks
  preserve sub-utterance speech and exclude merged silence and overlap padding.
  Duration constants derive from the evaluation sample rate. Frame counts use
  `math.ceil`, with the actual final-frame span retained for majority decisions.
  Primary transformations use separate assignments, now required by
  [AGENTS.md](../../AGENTS.md#code-and-checks).
- `prepare_reference` constructs reusable `ReferenceSpeech` timing and read-only
  activity masks for both reference channels. `prepare_separation` runs VAD only
  on present output channels and returns initial T/R `ChannelResult` records
  with VAD, utterances, and activity populated. Target-only outputs keep residual
  reference timing with unavailable residual activity. Other metrics remain
  disabled for Stage 4 to populate. Reuse is explicit and scoped by the caller
  to one mixture; there is no persistent or cross-recording cache.
- Added 38 offline CPU tests for merge/intersection and
  eligibility boundaries, fragmented overlap, padding, activity counts, fake
  Silero configuration/model loading, independent channel/recording state,
  reference reuse across inference variants, target-only output, and errors.
  The isolated package-import test now includes `vad.py` and still verifies
  that optional runtimes stay unloaded.

**Checks:** in the `audio` environment on Python 3.12.13,
`python -m ruff check .`, `python -m ruff format --check .`,
`python -m pytest` (155 tests after review), and `git diff --check` passed.
Pytest emitted the existing AnyIO plugin and three audioread deprecation warnings.
A separate, explicitly run CPU check loaded the installed packaged Silero 6.2.1
model and processed synthetic one-second and 200-sample silence independently;
both returned no speech. No downloads, source data, or GPU were used; no checks
were blocked.

**Deviations:** review removed defensive timestamp validation and duplicate
Silero state/inference handling. Thread configuration is left to users and
external libraries. CLI/pipeline wiring, requested-metric selection, and
recording-level VAD failure reporting remain in Stage 4.

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
sanity check that mixture-as-estimate gives zero improvement in an opt-in SI-SDR metric-backend
test. Start a synthetic CLI fixture with fake VAD/metrics, then extend it through
later stages. Cover shared resource preflight and failure isolation once.

**Implementation notes:**

- Added [`metrics.py`](../../src/realtime_speech_separation/evaluation/metrics.py)
  and [`pipeline.py`](../../src/realtime_speech_separation/evaluation/pipeline.py).
  Functional SI-SDR uses CPU float32 tensors and explicit `zero_mean=True`;
  provenance records the metric backend's epsilon and placement. Assumption:
  test exact centered silence in float64 without squaring, avoiding mean-rounding
  artifacts for constant float32 signals and underflow for quiet varying signals.
  Actual scoring retains TorchMetrics' float32 centering and stabilization.
  Reference zero energy is ineligible for SI-SDR; separated zero energy and
  mixture zero energy make their respective absolute scores undefined.
- Classical STOI passes reference first, at 16 kHz with `extended=False`.
  Assumption: pystoi's documented insufficient-STFT-frame warning means
  `ineligible/insufficient_stoi_frames`; identify it by warning text, never by the
  returned sentinel or a zero score. Other warnings retain normal handling.
  A missing baseline leaves valid absolute output scores intact, with undefined
  improvement for valid unavailability or failed improvement for a backend
  failure. Nonfinite scores fail locally; unexpected programming errors propagate.
- The pipeline slices all waveforms at identical reference bounds, preserves
  T/R identity, and reuses reference VAD and per-family/channel/interval mixture
  baselines across inference variants of one mixture. STOI/STOIi share calls, as
  do SI-SDR conditions when bounds coincide. Short unpaddable overlap candidates
  remain explicit ineligible records. Missing-utterance diagnostics inspect
  centered separated waveforms independently of separated VAD or metric selection.
- Assumption: `core` includes both `si_sdri` and `overlap_si_sdri`, plus `activity`,
  `stoi`, and `stoii`. Individual choices use schema metric names; combinations
  are deduplicated. `all` still means the full protocol and remains the default,
  so it currently fails with an explicit list of unimplemented metrics. Users
  must select `--metrics core` or implemented individual metrics until later
  stages land. All supported subsets are labeled partial evaluations in logs
  and persisted configuration. The README now shows the runnable core command.
- Preflight initializes each selected metric adapter and packaged Silero model
  once, before processing or writing results, with lazy optional imports, CPU
  execution, no downloads, and no RTSS thread configuration. Activity-only runs
  do not import metric runtimes. Reference and separated VAD run independently
  on every available channel, retaining `reference_vad_error` and
  `separated_vad_error` even when activity is disabled. A reference-channel
  failure leaves its speech populations unknown and prevents overlap construction;
  healthy reference channels still support intrusive scoring. Separated utterances
  remain available for future DNSMOS scoring despite reference VAD failures.
  Any VAD failure makes the invocation fail and prevents result reuse. Regression
  tests cover either reference channel failing across inference variants and both
  reference/separated errors persisting together with activity disabled.
  Metric selection is resolved once inside pipeline construction; `_score_channel`
  receives only reference overlap regions in addition to its prepared channel.
- Source SHA-256/path validity keys, native audio metadata, resolved settings,
  backend versions, and packaged VAD model digest now reach persistent results.
  Reuse delegates to Stage 1's structural/configuration/source checks. Stale,
  malformed, failed, and invalid results are recomputed; undefined/ineligible
  outcomes are reusable. `--overwrite` bypasses reuse. Cached-path bookkeeping
  belongs only to the current invocation and never modifies persisted results.
- The CLI now writes atomic per-separation JSON and pooled JSON/recording CSV
  reports with evaluation and selection IDs, including cached results reached
  through current discovery only. Full-set and filtered reports coexist. Invalid
  triples retain relative error paths and reasons, continue to later inputs,
  and contribute invalid counts. Empty selections log a diagnostic; known hashes
  still get reports for all-invalid or empty-candidate runs. Configuration,
  file-I/O, and requested backend failures return nonzero; valid mathematical
  unavailability does not. The legacy rate flag only accepts 16000 and permutation
  matching is rejected.
- Added explicit failed `MetricResult` outcomes for missing VAD prerequisites.
  Both in-memory and persisted overlap use `None` for reference VAD failure
  and an empty tuple for successful VAD without overlap. Each channel/metric summary now includes
  `recording_coverage`: `candidate_recordings`, `scored_recordings`,
  `no_candidate_recordings`, `excluded_due_to_errors`,
  `partially_failed_recordings`, and `exclusion_reasons`. Activity and WER expose
  the same recording accounting. All-interval failures count as full error
  exclusions; partial failures retain surviving scores. Reason codes exclude
  runtime detail after `: ` and count each excluded recording once per distinct
  reason. Disabled/missing outputs do not contribute candidates; invalid inputs
  remain in the separate recording totals. Recording coverage stays in summary
  JSON; CSV rows retain interval/frame/word coverage and metric-level failure
  reasons without per-recording binary coverage fields. Future adapters must mark failed
  prerequisites explicitly rather than fabricate empty candidate populations.
  Tests cover all nine reference/output T/R VAD failure combinations, successful
  empty populations, undefined scores, full/partial interval errors, missing
  outputs, failed-result round trips/reuse, and activity/ASR exclusion reasons.
- Naming assumption: each directory has unique mixture stems and each mixture
  has at most one separated audio file per inference suffix, regardless of
  extension. Ambiguities are rejected to prevent result-file collisions. Result
  files use the mixture stem directly rather than replacing marker text anywhere
  in a path. CSV column order stays fixed between fresh and cached runs.
- Replaced Asteroid/pandas in `eval` with explicit `torch>=2.2`,
  `torchmetrics>=1.8.2`, `pystoi>=0.4.1`, and the existing Silero requirement.
  TorchMetrics 1.8.2 requires Python >=3.9; pystoi 0.4.1 supplies a Python wheel
  without a narrower Python requirement. Both were checked on local Python 3.12;
  base/dev and unrelated extras are unchanged. References:
  [functional SI-SDR](https://github.com/Lightning-AI/torchmetrics/blob/v1.8.2/src/torchmetrics/functional/audio/sdr.py),
  [pystoi warning/API](https://github.com/mpariente/pystoi/blob/v0.4.1/pystoi/stoi.py).
- Replaced the legacy CLI fixture with synthetic end-to-end offline tests using
  injected resources. These cover channel assignment, exact slices and paired
  scores, baseline/resource reuse, cache invalidation/overwrite, filtered cached
  reports, target-only output, missing/silent audio, zero candidate populations,
  short overlap eligibility, unavailable metric selections, preflight failures,
  and failure isolation/retry. Adapter tests cover argument order, float32 SI-SDR,
  centered silence/quiet output, real zero STOI, warning mapping, nonfinite
  scores, unavailable baselines, and visible programming errors. The isolated
  import test now includes both new modules. The real SI-SDR test is CPU-only and
  opt-in via `RTSS_TEST_METRIC_BACKENDS=1` and pytest's `metric_backend` marker.

**Checks:** in the `audio` environment on Python 3.12.13,
`python -m ruff check .`, `python -m ruff format --check .`,
`python -m pytest` (199 passed, one opt-in test skipped), and `git diff --check`
passed. The skipped real CPU SI-SDR sanity test passed when explicitly enabled
with `RTSS_TEST_METRIC_BACKENDS=1` and `-m metric_backend`. A temporary synthetic
target-only silence triple passed actual-resource core CLI evaluation, strict
JSON reporting, and cached rerun. A separate installed pystoi call confirmed the
short-input warning mapping. Runtime versions were TorchMetrics 1.8.2, pystoi
0.4.1, PyTorch 2.13.0, Silero 6.2.1, NumPy 2.4.6, and SciPy 1.17.1. Pytest emitted
the existing AnyIO plugin and three audioread deprecation warnings. No source
data, enrollment, metadata, separator resources, model-asset downloads, or GPU
were used. No checks were blocked.

**Deviations:** basic report writing, empty-run handling, and filtered-cache
integration checks were brought forward from Stage 10 so the replacement CLI is
usable and reviewable now. Full-protocol integration, additional metric adapters,
and remaining output-consumer/native-setup documentation stay in their owning
stages. The run-eval skill now documents metric selection and the result/report
formats. No metric definitions changed.

### Stage 5 — Add PESQ

- Add wideband PESQ over reference utterances and include it in `eval`.
  Document native setup and resolved Python/NumPy compatibility.
- Map documented no-utterance/too-short exceptions to interval-result reasons using
  the selected exception mode.

**Tests:** verify PESQ arguments and exception mapping with a fake PESQ metric backend.
An opt-in wrapper smoke test checks a valid signal and documented error cases.

**Implementation notes**

- `--metrics pesq` scores the separated and reference channel at identical
  reference-utterance bounds using 16 kHz wideband mode and
  `PesqError.RAISE_EXCEPTION`. `--metrics core pesq` combines PESQ with core
  metrics. PESQ settings and the installed wrapper version are part of the
  evaluation ID. Tests substitute the PESQ module, including its scoring function
  and exception types. The existing interval reporting and cache rules apply.
- The implementation assumes `BufferTooShortError` and `NoUtterancesError` mean
  the interval is ineligible for PESQ, with reasons `pesq_buffer_too_short` and
  `pesq_no_utterances`. They retain a candidate with a null value. Other typed
  PESQ exceptions are scoring failures. Version 0.0.4 can also raise
  `ValueError` when it tries to convert a native NaN to an integer error code,
  as observed with silent and very quiet estimates. That exact error uses the
  same `failed/nonfinite_score` result as a nonfinite return. Other `ValueError`s
  propagate. [PESQ exception implementation](https://github.com/ludlows/PESQ/blob/v0.0.4/pesq/cypesq.pyx)
- The `eval` extra sets `pesq>=0.0.4`. PyPI publishes the tested release as source
  with no Python version constraint. An isolated build succeeded on Linux with
  CPython 3.12.13, and its wheel passed the native
  smoke test with NumPy 1.26.4 and 2.4.6. Pip resolved NumPy and Cython wheels for
  the CI versions, Python 3.11 and 3.14. Native execution was verified on Python
  3.12. [PyPI release](https://pypi.org/project/pesq/0.0.4/)
  [Native build requirements](https://github.com/ludlows/PESQ/blob/v0.0.4/setup.py)
  [Installation and CLI usage](../../README.md#pesq-setup)

**Checks**

Ruff lint and formatting, `git diff --check`, and pytest passed on Python 3.12.13
(207 passed, two opt-in tests skipped). Both opt-in metric tests passed when
enabled. A CPU CLI smoke test on synthetic speech verified per-channel PESQ,
JSON and CSV reports, cache reuse, and preservation of the residual score when
the silent target estimate produced a nonfinite result and a nonzero exit code.

### Stage 6 — Add ViSQOL

- Initialize the official API in 16 kHz speech mode with a local speech model;
  return per-utterance MOS-LQO and document native setup.
- Establish the pinned ViSQOL metric backend's actual short-input requirements.
  Apply the ViSQOL input duration and padding rules from the
  [protocol](../evaluation-protocol.md#pesq-stoi-visqol).
- Investigate the official CLI only if the Python API cannot run in the target
  environment; record that decision before adding a fallback adapter.

**Tests:** verify speech-mode/model configuration with a fake ViSQOL metric backend. Use an
opt-in smoke test to establish native short-utterance behavior.

**Implementation notes**

- `--metrics visqol` loads the official Python API once on CPU. The adapter
  passes the aligned, padded reference and separated utterance prefixes as
  contiguous float64 arrays and records `moslqo`. Assumption: use the official
  bundled 16 kHz speech lattice model at its installed path. RTSS enables the lattice
  explicitly and uses the upstream search radius of 60 and native alignment
  defaults. No runtime download is needed. The evaluation ID includes the
  binding and Protobuf versions, the extension SHA-256, the speech model
  SHA-256, and the resolved scoring settings.
- At Stage 6 the evaluation rule used utterance span duration. Spans under two
  seconds are `ineligible/visqol_buffer_too_short`. The first nine seconds at
  most are scored with 0.5 seconds of zero padding at each end of both signals.
  The padding and span limits are derived from the sample rate and included in
  the evaluation ID. Results retain the original reference utterance bounds.
  Native status, runtime, and indexing errors during scoring are interval
  failures, as are nonfinite scores.
  [Upstream input guidelines](https://github.com/google/visqol/blob/master/README.md#guidelines)
- Pinned official revision `38d0b0163e441047d4429bf07ad09e5b9031d02c`, whose
  installer reports version 3.3.3. The setup uses that revision's Bazel 5.3.2,
  updates pybind11 to 2.11.2 for Python 3.12, and supplies `<cstdint>` when
  compiling the pinned TensorFlow Lite source with GCC 13. Protobuf 3.20.3
  and 5.29.6 worked on Python 3.12. Protobuf 7.34.1 failed configuration because
  the pinned binding uses `MessageFactory.GetPrototype`, which was removed in
  version 6. RTSS surfaces configuration failures during preflight. The
  Python API worked on the target host, so the implementation uses it directly.
  The local model SHA-256 is
  `4bd031d6c95594ae2cd1d471aa6611bff26b3bbf816528431ff1741bcd798dc6`.
  [Upstream installer](https://github.com/google/visqol/blob/38d0b0163e441047d4429bf07ad09e5b9031d02c/setup.py)
  [Protobuf migration guide](https://protobuf.dev/support/migration/)
  [Native setup](../../README.md#visqol-setup)

**Checks**

Ruff lint and formatting, `git diff --check`, and pytest passed in the `audio`
environment (Python 3.12.13, 214 passed, three opt-in tests skipped). All three
opt-in CPU metric tests passed when explicitly enabled with
`RTSS_TEST_METRIC_BACKENDS=1`, `RTSS_TEST_VISQOL=1`, and `-m metric_backend`.
Native build and execution were verified on Linux x86-64 with Python 3.12.13
and NumPy 2.4.6. Python 3.14 support was added in pybind11 3.0, which is newer
than the build dependency verified here. The Protobuf 5 runtime emitted the
expected `MessageFactory` deprecation warning.
[pybind11 changelog](https://pybind11.readthedocs.io/en/stable/changelog.html)

Boundary tests verify two-second eligibility, the nine-second cap, and the
exact zero padding in both inputs. On temporary copies of upstream test speech,
the CLI reported one scored
utterance and one ineligible utterance with its original 8,464-sample span,
wrote JSON and CSV, and reused both results on the second run.

### Stage 7 — Add DNSMOS-P.808

- Add the TorchMetrics adapter and CPU ONNX Runtime/requests to `eval`.
  Select only `p808_mos`, preserving the library's short-input extension and
  internal-window averaging rather than adding an utterance-duration cutoff.
- Document and checksum all assets needed by the selected API, including any
  companion models. Provision its `~/.torchmetrics/DNSMOS` cache for offline use;
  the current API has no public model-path argument. Reconsider the adapter only
  if that location cannot meet deployment needs.

**Tests:** a fake four-component return verifies P.808 selection and separated
utterance bounds. An opt-in DNSMOS metric-backend smoke test covers short and long utterances
with pre-provisioned assets.

**Implementation notes**

- `--metrics dnsmos_p808` uses the separated VAD utterances on each available
  channel. It sends each complete 16 kHz span to the CPU TorchMetrics function
  and retains its first component, `p808_mos`. TorchMetrics handles short-input
  repetition and internal window averaging. Failed separated VAD marks that
  channel's DNSMOS measurement as failed. Reference VAD failures leave scores
  from healthy separated channels available. A successful VAD with zero eligible
  utterances yields zero candidates.
- Assumption for provisioning is a stable, readable TorchMetrics model cache
  for the evaluation account. The two files are Microsoft's `model_v8.onnx`
  (P.808) and `sig_bak_ovr.onnx` (companion for the non-personalized call).
  The setup pins Microsoft DNS-Challenge revision
  `785ab12275d5cf9341c5e946b0eee9f8825ece3b` and verifies both SHA-256 hashes.
  RTSS fingerprints both installed files before loading either session and
  includes those digests, scoring settings, and runtime versions in the
  evaluation ID. Preflight uses TorchMetrics' private cached session loader
  because its public scoring API loads the sessions lazily. It refreshes the
  cache on pipeline creation so subsequent invocations in the same process
  can pick up replaced models. The CPU sessions are shared by all scores in
  the run. Missing or invalid assets fail preflight with setup instructions.
  [Cache setup and checksums](../../README.md#dnsmos-p808-setup) ·
  [Microsoft ONNX assets](https://github.com/microsoft/DNS-Challenge/tree/785ab12275d5cf9341c5e946b0eee9f8825ece3b/DNSMOS/DNSMOS) ·
  [TorchMetrics cache and scoring implementation](https://github.com/Lightning-AI/torchmetrics/blob/v1.8.2/src/torchmetrics/functional/audio/dnsmos.py)
- The `eval` extra adds `onnxruntime>=1.20.1` and `requests>=2.32.5`.
  TorchMetrics requires requests for scoring even when the ONNX assets are
  provisioned offline. PyPI publishes Python 3.11 and 3.12 CPU wheels for ONNX
  Runtime 1.20.1 and Python 3.14 CPU wheels for newer releases within the range.
  [ONNX Runtime 1.20.1 files](https://pypi.org/project/onnxruntime/1.20.1/#files) ·
  [ONNX Runtime 1.32.0 files](https://pypi.org/project/onnxruntime/1.32.0/#files)

**Checks**

Ruff lint and formatting, `git diff --check`, and pytest passed in the
`audio` environment on Python 3.12.13 (222 passed, four opt-in tests skipped).
The opt-in test passed with `RTSS_TEST_DNSMOS=1` and `-m metric_backend` using
TorchMetrics 1.8.2 and CPU ONNX Runtime 1.20.1 and 1.32.0. It blocked the
download path and verified that the two sessions served both a 0.2-second
input and a 13-second input.

The cache was provisioned for the local evaluation account. A CLI smoke test
on temporary copies of two public test-speech triples verified a scored
separated utterance, a recording with no eligible separated utterances,
JSON and CSV reporting, and reuse of both results.

#### Follow-up on metric activity and settings

Each interval adapter declares `min_active_samples` in its settings. Preflight
rejects adapter definitions that omit it. PESQ and STOI declare the standard
VAD utterance minimum. ViSQOL requires three seconds of reference activity,
and DNSMOS requires three seconds of separated activity. The check uses original
VAD activity, excluding merged gaps and padding. Under-threshold candidates retain their bounds and are
reported as `ineligible/insufficient_activity`. The shared SI-SDRi adapter
uses the 0.2-second overlap minimum. Ordinary SI-SDRi candidates still come
from utterance extraction, which requires 0.5 seconds of activity. Eligible
overlaps are padded to a half-second span when needed. The loader rejects
files shorter than one second after resampling, so accepted recordings
always have room for the required overlap padding.

The evaluation ID includes the recording length minimum and each selected
interval adapter's activity minimum. The adapters use their recorded parameters
to configure and invoke their backends. The pipeline invokes all interval
adapters through a shared path and passes a reference waveform only when the
adapter declares `intrusive`. ViSQOL and DNSMOS also reject direct calls with
waveform spans too short to contain the required activity.
See the [evaluation protocol](../evaluation-protocol.md).

Ruff lint and formatting, `git diff --check`, and pytest passed
(234 passed, four opt-in tests skipped). A CPU CLI smoke test selected both
ViSQOL and DNSMOS on temporary copies of test speech. Both metrics reported a
shorter utterance as ineligible and scored a longer utterance. The rerun reused
both results.

### Stage 8 — Add WeSpeaker similarity

- Add `speaker.py` for the selected SimAMResNet34 ONNX model and local model path.
  Reuse the official filterbank frontend, amplitude scaling, frame settings, and
  mean normalization. Include compatible TorchAudio and CPU ONNX Runtime in `eval`.
- Implement the protocol's reference-prototype and separated-utterance rules,
  including the agreed embedding average. Report unavailable prototypes,
  zero-norm embeddings, and extraction failures.
- Return raw cosine in `[-1, 1]`, not WeSpeaker's convenience `[0, 1]` transform.
  Preserve upstream attribution when adapting its inference example.

**Tests:** deterministic fake embeddings verify per-recording/channel scope,
equal weighting, detected speech eligibility at the 3-second boundary,
missing prototypes, and negative cosine. Before accepting this frontend, compare
embeddings with the official extractor using the same model/configuration in an
opt-in local check.

**Implementation notes**

- `--metrics speaker_cosine` defaults to a revision-specific model cache under
  `${XDG_CACHE_HOME:-$HOME/.cache}/rtss/wespeaker/`. A missing or corrupt entry
  triggers a standard-library HTTPS download from ModelScope. The download is
  verified before it is published atomically, and an optional
  `--speaker_model_path` bypasses the cache. Preflight fingerprints the model
  and checks for the variable-length float32 `feats` input and 256-dimensional
  `embs` output. The model's SHA-256, frontend and eligibility settings, and
  runtime versions are included in the evaluation ID. The documented asset is
  the VoxBlink2-only export at ModelScope dataset revision
  `548e87a3fa9fb727a7d08d1387e1f710093c56dd`, SHA-256
  `1564bb8490c7f7759e293bbc2b2d326b5dc2b79f24e47a0d57f4a24064a58ad9`.
  [CLI and cache setup](../../README.md#wespeaker-setup) ·
  [WeSpeaker model catalog](https://github.com/wenet-e2e/wespeaker/blob/9fecd6cb4f47475d01761d87c826298dff4ef18c/docs/pretrained.md)
- The adapter follows the official ONNX example using 16 kHz float32 mono
  audio, waveform scaling by 32768, 80 Kaldi mel bins, 25 ms Hamming windows
  with a 10 ms shift, zero dither, no energy feature, and utterance-level mean
  subtraction. One CPU ONNX session serves the invocation and uses runtime
  thread defaults. The frontend's upstream attribution and license are shipped
  with the package. `eval` uses `torch>=2.11` and `torchaudio>=2.11` for
  TorchAudio's stable PyTorch ABI and includes CPU ONNX Runtime
  (`onnxruntime>=1.20.1`). TorchCodec is unnecessary for the
  in-memory frontend. [Official ONNX example](https://github.com/wenet-e2e/wespeaker/blob/fb2325c1769958ccf979667862193419c9fc0551/wespeaker/bin/infer_onnx.py) ·
  [TorchAudio compatibility](https://docs.pytorch.org/audio/main/installation.html)
- Reference embeddings use complete merged utterance spans and require at
  least 48,000 raw VAD-active samples. Each reference channel's raw vectors
  receive equal weight before normalizing their mean. Both reference outcomes
  are reported for target-only output. Prototypes are shared across one
  mixture's variants. The non-intrusive `SpeakerCosine` metric uses a
  channel-bound prototype and the shared interval scoring loop. Separated
  candidates also require 48,000 raw VAD-active samples. A merged gap counts
  toward the waveform span passed to the extractor and is excluded from the
  eligibility calculation. The activity requirement replaces the previous
  span threshold in the evaluation identity so cached span-based results
  are recomputed.
- Reporting assumptions are that an eligible separated candidate with an
  unavailable prototype is undefined. A failed reference VAD or required
  extraction marks eligible separated candidates as failed. Short candidates
  retain their ineligibility. A zero-norm reference embedding makes the
  prototype unavailable, as does a zero-norm mean. A zero-norm separated
  embedding is undefined, and nonfinite embeddings are extraction failures.
  Healthy utterances and channels retain their scores. Results with backend
  failures are recomputed on the next invocation.

After running a speaker evaluation to populate the cache, copy the model and
the pinned official `infer_onnx.py` into a temporary directory for the opt-in
frontend comparison.

```bash
wespeaker_assets=$(mktemp -d)
wespeaker_cache="${XDG_CACHE_HOME:-$HOME/.cache}/rtss/wespeaker/548e87a3fa9fb727a7d08d1387e1f710093c56dd"
cp "$wespeaker_cache/voxblink2_samresnet34.onnx" "$wespeaker_assets/"
curl --fail --location --output "$wespeaker_assets/infer_onnx.py" \
    https://raw.githubusercontent.com/wenet-e2e/wespeaker/fb2325c1769958ccf979667862193419c9fc0551/wespeaker/bin/infer_onnx.py
RTSS_TEST_WESPEAKER=1 RTSS_WESPEAKER_TEST_ASSETS="$wespeaker_assets" \
    python -m pytest tests/test_evaluation_speaker.py -m metric_backend
```

**Checks**

Ruff lint and formatting, `git diff --check`, and pytest passed in the `audio`
environment on Python 3.12.13 (246 passed, five opt-in tests skipped by default).
The opt-in test matched the official extractor's 256-dimensional output within `1e-5` absolute
and relative tolerances for spans of 48,000 and 64,321 samples. It used identical
decoded samples with PyTorch 2.13.0, TorchAudio 2.11.0, and CPU ONNX Runtime
1.20.1 and 1.32.0. The test supplies WAV decoding through SoundFile so the
official frontend can run with no TorchCodec dependency.

On temporary copies of two public speech triples, the CPU CLI reported one
scored target utterance and one short, ineligible utterance. It wrote both
JSON and CSV reports and reused both results on rerun. An invalid ONNX file
failed preflight with a nonzero exit status before writing any reports.

After review, pytest passed with 248 tests and five opt-in skips. The opt-in
WeSpeaker parity check passed separately. The CPU CLI populated an empty
temporary cache from ModelScope and reused the verified asset and both results
on rerun with HTTPS routed to an unreachable local proxy.

After correcting the eligibility rule, the CLI test verifies that 48,000
active samples qualify and that 47,999 active samples in a 48,000-sample span
are ineligible. The full suite still passes. A CPU CLI rerun recomputed both
results produced under the previous span rule and wrote a new report.

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
- Document user-controlled CPU parallelism in the README, including runtime
  environment variables such as `OMP_NUM_THREADS` set before launching evaluation.
  RTSS does not explicitly configure thread counts. Check and describe backend
  overrides: Silero 6.2.1 itself calls `torch.set_num_threads(1)` during import,
  so environment variables alone do not override that backend-imposed setting.

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
Keep ordinary tests offline and CPU-only with injected metric, VAD, speaker, and
ASR backends and temporary synthetic audio; real model/native metric-backend tests
are opt-in. Record checks that
could not run, and keep runtime stacks out of base/dev CI.

For the final real-data smoke test, copy a few existing triples to a temporary
location and use `--input_filter` and `--overwrite` there. Verify evaluation works
without the project's `data` tree or enrollment files. Follow repository GPU
etiquette for any explicitly requested GPU check; evaluation testing does not
require separator inference or source-data changes.
