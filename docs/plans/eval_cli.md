# Evaluation CLI implementation plan

Status: planning only. No implementation, dependency installation, checks, or tests
were performed for this plan.

## Objective and scope

Implement the [evaluation protocol](../evaluation-protocol.md) through
[`rtss.eval`](../../src/realtime_speech_separation/eval_cli.py). The protocol remains
the source of metric definitions and motivations; this document describes the
implementation delta, dependency choices, interfaces, and staged delivery.

Keep new CLI usage instructions in this plan or usage documentation. Preserve
the protocol's existing level of detail, including its file conventions, audio
shapes, loading examples, and output requirements.

Evaluation must work with just the supplied mixture, reference, and separation
WAVs plus locally available evaluation models and runtime libraries. Run Silero directly
on each reference and separated channel. Do not read `data/vads`, source recordings,
dataset metadata, enrollment recordings, MagiCodec, or separator model resources.
Model provisioning is a separate installation concern: portable evaluation does
not mean that pretrained weights can be eliminated.

Preserve recursive discovery, input filtering, multiple inference configurations,
resumability, per-separation JSON, and JSON/CSV summaries. Support a mono separated
recording as target-only output. Never reconstruct a residual from the mixture or
choose a channel permutation.

## Current implementation and required delta

The relevant current files are
[`eval_cli.py`](../../src/realtime_speech_separation/eval_cli.py),
[`data_utils.py`](../../src/realtime_speech_separation/utils/data_utils.py),
[`pyproject.toml`](../../pyproject.toml), and
[`test_package.py`](../../tests/test_package.py).

| Area | Current behavior | Required work |
| --- | --- | --- |
| Discovery | Recursively finds mixtures, applies substring filters, finds adjacent separated files, groups by inference hash. | Retain this flow. Require separated matches to be audio files, parse the suffix relative to the mixture stem, sort deterministically, and report incomplete triples. The current prefix-only match can select non-audio sidecars. |
| Audio loading | Loads float32 through librosa, resampling to a configurable rate, default 16 kHz. Also loads an unused enrollment WAV. | Fix protocol evaluation to 16 kHz, remove enrollment loading, validate shapes, finite values, nonempty audio, and matching resampled lengths. Handle mono separated output explicitly. |
| Alignment | Assumes alignment and compatible lengths without explicit validation. | Validate lengths and document alignment as an input requirement. Inference already removes its configured delay; evaluation must not compensate again. |
| VAD | No evaluation VAD, despite `silero-vad` already appearing in the eval extra. | Load Silero once and score reference T/R and separated T/R independently. Preserve raw intervals and derive utterances, overlap regions, and activity frames. |
| SI-SDR | Asteroid scores full recordings and computes an output-minus-input `si_sdr_delta`. | Compute explicitly named SI-SDRi on each reference utterance and each reference overlap interval, separately for T/R, with the identical mixture slice as baseline. |
| Channel assignment | `--compute_permutation` can optimize assignment. | Reject permutation matching for this enrollment-conditioned protocol. |
| Activity | Absent. | Construct 100 ms masks from raw VAD; persist TP/FP/FN and derive P/R/F1. |
| Speaker similarity | Absent. | Extract multilingual WeSpeaker embeddings; build separate reference prototypes; score eligible separated utterances against the corresponding prototype. |
| Intrusive quality | Full-recording STOI only. Input STOI and STOI improvement are also produced. | Evaluate PESQ, standard STOI, STOI improvement, and speech-mode ViSQOL on reference-defined utterances, retaining matching mixture baselines. |
| Nonintrusive quality | Absent. | Score DNSMOS-P.808 on separated-defined utterances. |
| ASR/WER | Absent. | Transcribe complete reference/separated channels with whisper.cpp large-v3-turbo through pywhispercpp, then retain JiWER edit counts. |
| Per-sample results | Flat arrays of channel scores; no intervals, counts, eligibility, or configuration identity. | Introduce a channel-explicit schema containing raw counts and per-utterance records, with provenance and failure reasons. |
| Aggregation | Mean, SD, and requested percentiles already exist, but operate on fixed-size recording vectors. An entire recording is discarded if any top-level value is `None`. | Pool utterance scores across recordings, independently per channel and metric. Sum activity/WER counts before computing ratios. Report metric-specific coverage and failures. |
| Empty/failing cases | Indexes the first retained result, which fails when all results are excluded. Nonfinite values are not explicitly handled. | Handle empty discovery, no speech/overlap, missing residuals, partial metric failures, and empty distributions without crashing or writing NaN/Infinity JSON. |
| Cached results | Any nonempty existing JSON is reused unless `--overwrite` is given. | Require valid result structure, matching evaluation configuration, and source audio identities. |
| Summary selection | Only results reached through filtered mixture discovery enter the summary. | Preserve this behavior explicitly, including reused results. Give each filter selection its own report identity so dataset reports and overall reports coexist. |
| CSV | One row per accepted separated file, assuming every value is a channel vector. | Retain one row per separation, with explicit channel/metric summary and coverage columns; add an utterance table for variable-length observations. |
| Tests/documentation | Tests cover installation and entry points, with no evaluation behavior tests. | Add focused offline tests as stages are implemented; document resource setup, selected metrics, schema, and portability. |

## Library and resource decisions

These are recommended implementation choices based on upstream documentation and
source reviewed for this plan. During the relevant stage, resolve and record
compatible released versions for local Python 3.12 and the package's supported
Python versions. No dependency resolution has been attempted here.

| Component | Recommended implementation | Dependency and integration considerations |
| --- | --- | --- |
| Audio I/O and resampling | Existing librosa, NumPy, and soundfile. | Reuse the existing stack and preserve float32 waveform amplitudes. Avoid another audio abstraction. |
| VAD | `silero_vad.load_silero_vad` and `get_speech_timestamps`, CPU initially. | Already declared in the eval extra. Use the packaged model and array input, avoiding Torch Hub downloads and redundant audio decoders. Upstream supports passing a waveform without its optional audio I/O extras. [Silero source](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py). |
| SI-SDR | TorchMetrics functional `scale_invariant_signal_distortion_ratio`. SI-SDRi is a subtraction of two calls. | Use the functional API to retain individual observations. Set zero-mean behavior explicitly. Torch is already needed by Silero; TorchMetrics avoids retaining Asteroid's separation/training stack solely for scoring. [SI-SDR API](https://lightning.ai/docs/torchmetrics/stable/audio/scale_invariant_signal_distortion_ratio.html), [base dependencies](https://github.com/Lightning-AI/torchmetrics/blob/master/requirements/base.txt). |
| STOI | `pystoi.stoi(reference, estimate, 16000, extended=False)`. | A focused implementation using the existing scientific Python stack. Observe warnings/errors on insufficient speech rather than accepting its failure sentinel as a real observation. [Usage](https://github.com/mpariente/pystoi), [implementation](https://github.com/mpariente/pystoi/blob/master/pystoi/stoi.py). |
| PESQ | `pesq.pesq(16000, reference, estimate, mode="wb")`, exception mode. | Native extension: check Python/NumPy compatibility and build prerequisites in its stage. Convert documented buffer/no-utterance errors into metric-local failures, never a valid negative score. [PESQ wrapper](https://github.com/ludlows/PESQ). |
| ViSQOL | Official Google ViSQOL Python API in speech mode, with its corresponding model asset. | Requires a native build/install and model provisioning; avoid a TensorFlow Python dependency solely to invoke it. Prefer the reusable array API over spawning one process per utterance. Confirm native dependencies and Python compatibility in its stage. [Official installation/API](https://github.com/google/visqol#python-api-usage). |
| DNSMOS-P.808 | TorchMetrics `deep_noise_suppression_mean_opinion_score`, `personalized=False`, selecting the P.808 component. | Add CPU `onnxruntime` and `requests` explicitly. Upstream returns P.808 plus three P.835 scores, repeats short inputs for its analysis window, and caches sessions. Preserve that behavior and retain only P.808 for the protocol. [Implementation/API](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/dnsmos.py). |
| Speaker embeddings | Official WeSpeaker multilingual VoxBlink2 SimAMResNet34 ONNX model, with the official inference frontend. | Prefer a small adapter following WeSpeaker's documented ONNX deployment path, using `onnxruntime` and `torchaudio.compliance.kaldi` for features. Pin matching torch/torchaudio builds. This reuses the toolkit's model/frontend without its full training and diarization installation. [Model list](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md?plain=1), [ONNX inference](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/infer_onnx.py). |
| Transcription | In-process `pywhispercpp.model.Model`, using local `ggml-large-v3-turbo.bin` and 16 kHz float32 NumPy input. | Reuse a loaded model and consume returned segments. No separate executable, temporary WAV, FFmpeg, Transformers, or remote ASR service is needed for array input. Pin the binding/native build and verify CPU wheels on supported platforms. [Binding API](https://github.com/absadiki/pywhispercpp/blob/main/pywhispercpp/model.py), [installation](https://github.com/absadiki/pywhispercpp#installation). |
| WER | `jiwer.process_words`, retaining substitutions, deletions, insertions, and reference length. | Reuses RapidFuzz edit-distance code. Use JiWER 4+ for defined handling of empty references; apply the protocol's explicit aggregate denominator policy. [JiWER](https://github.com/jitsi/jiwer), [usage](https://jitsi.github.io/jiwer/usage/). |
| Activity counts, interval operations, cosine, aggregation | Small NumPy/Python helpers. | These are trivial operations or protocol-specific orchestration. Do not implement signal metrics, feature extraction, neural inference, or edit-distance algorithms from scratch. |

Other suggested packages were considered:

- **Asteroid:** currently used, and its metric wrapper supports more than the two
  selected metrics, including PESQ. It does not supply the protocol's segmentation
  and aggregation. Replace its use once the core adapters are integrated. Its
  declared dependencies include PyTorch Lightning, filterbanks, audio utilities,
  optimizers, and other separation-related packages.
  [Wrapper](https://github.com/asteroid-team/asteroid/blob/master/asteroid/metrics.py),
  [dependencies](https://github.com/asteroid-team/asteroid/blob/master/setup.py).
- **descript-audiotools:** useful as a general audio framework, but introduces
  additional notebook, plotting, logging, and audio utilities when this project
  already has I/O/resampling. Use focused metric packages instead.
  [Dependencies](https://github.com/descriptinc/audiotools/blob/master/setup.py).
- **speechonnxmetrics:** a plausible future alternative for P.808, with NumPy,
  ONNX Runtime, and Hugging Face Hub as its base dependencies and a dedicated
  `dnsmos_p808` metric. Its README currently describes installation from Git.
  Prefer the already selected TorchMetrics implementation initially; reconsider
  if its local-model/cache limitations cannot meet the deployment contract.
  Any adoption must pin a release or commit and verify P.808 behavior against the
  upstream implementation. Do not adopt its alternate ASR/VAD/speaker defaults.
  [README](https://github.com/TigreGotico/speechonnxmetrics),
  [dependency manifest](https://github.com/TigreGotico/speechonnxmetrics/blob/dev/pyproject.toml).
- **mir_eval / museval:** no additional BSS Eval SDR/SIR/SAR metrics are requested.
  Their distortion-filter scoring must not be substituted for SI-SDR.
  [mir_eval](https://github.com/mir-evaluation/mir_eval),
  [museval's BSS Eval description](https://github.com/sigsep/sigsep-mus-eval).
- **Full WeSpeaker Python installation:** keep out of the core eval extra. Its
  current package manifest includes diarization and representation-learning
  dependencies, including s3prl, Python Whisper, PEFT, and Accelerate. Use the
  documented ONNX inference path above; if that path fails parity validation,
  evaluate the official Python extractor as an explicitly heavier alternative.
  [Package dependencies](https://github.com/wenet-e2e/wespeaker/blob/master/setup.py).

Suggested extras are `eval` for Silero/TorchMetrics/pystoi, `eval-quality` for
PESQ/DNSMOS dependencies, `eval-speaker` for the ONNX speaker frontend, and
`eval-asr` for pywhispercpp and JiWER. Have the latter extras include `eval`.
Document ViSQOL native setup and pywhispercpp wheel/build options separately.
Keep all of these outside the base/dev
dependency sets; avoid broad `torchmetrics[audio]` or toolkit “all” extras.
Keep pandas initially for existing reports, then remove it from `eval` only if
the rewritten CSV writer no longer needs it. Do not change unrelated extras.

## Contracts to settle in Stage 1

The following are proposed defaults where the protocol leaves implementation
details open. They are not new claims about what it already specifies. Record
accepted metric semantics in the protocol, and implementation/resource details
in this plan or usage documentation. Later stages should not quietly choose
different policies.

1. **Audio:** evaluate at 16 kHz only. Keep `--sample_rate=16000` as a compatibility
   alias and reject other values. Accept different native rates if resampled
   lengths agree exactly. Reject mismatched lengths instead of automatically
   shifting, truncating, or zero-padding complete recordings. Temporal alignment
   remains a caller guarantee; equal lengths alone cannot prove it. Do not
   normalize loudness, peak-normalize, or clip the common waveform.
2. **Channels:** mixture is loaded mono; reference must contain exactly T/R;
   separated output may contain T alone or T/R. A mono separation is always T.
   Accept target-only arrays as `(n,)` or `(1, n)`, normalizing to `(1, n)`.
   Reject other channel counts. Reference R still defines overlap for target-only
   outputs, even though residual output metrics are unavailable.
3. **Raw VAD:** use integer sample indices and explicit parameters, proposed as
   threshold 0.5, negative threshold 0.35, minimum silence 100 ms, no timestamp
   padding, and no minimum speech-duration removal inside Silero. “Raw” means
   Silero timestamps before the protocol's merging/filtering. This avoids
   silently losing short activity before activity/overlap analysis. Pin the
   model version and record parameters rather than inheriting changing defaults.
4. **Intervals:** use sorted, disjoint half-open sample intervals `[start, end)`.
   Merge gaps of at most 1 second. Track original active samples independently
   of merged interval span; merged gaps do not become speech. Apply the protocol's
   inclusive minimum-duration and active-duration thresholds in samples.
5. **Overlap:** derive evaluation overlap from raw reference T/R intersections.
   Merge/filter before padding. Preserve actual overlap duration separately from
   padded span. At recording boundaries, move unavailable padding to the other
   side while staying inside the recording. If a recording is too short to
   provide a 0.5-second span, mark the interval ineligible with a reason.
6. **Activity:** strict majority means more than 50% of a frame; a tie is inactive.
   For the final partial frame use its actual duration as the denominator.
   An undefined P/R/F1 denominator produces `null`, with raw counts retained.
7. **SI-SDR:** set `zero_mean=True` explicitly and record backend/dtype/epsilon
   behavior. A zero-energy reference slice is ineligible. Guard zero-energy
   estimates and mixture baselines after centering: their SI-SDR is undefined,
   so retain a `null` score with an explicit reason instead of accepting the
   backend's stabilized value. Retain the observation and a silent-estimate
   diagnostic in coverage. Do not classify a quiet but nonzero estimate as
   undefined merely because Silero finds no speech. Other metric adapters still
   evaluate the same reference slice when their definitions allow it.
8. **Speaker prototypes:** use only reference utterances containing at least
   3 seconds of raw VAD-active speech, within one channel of one evaluation
   recording. Merged silent gaps cannot satisfy this requirement. Do not pool
   unrelated speakers across the dataset. Average the eligible raw embeddings
   with equal utterance weight, then use ordinary cosine normalization. If no
   reference utterance qualifies, that channel's prototype and similarity are
   unavailable; other metrics remain computable. The separated similarity
   population keeps the protocol's 3-second utterance-span minimum. An unavailable
   required reference embedding makes that channel's prototype unavailable.
   Residuals containing several speakers retain the protocol's one-prototype
   interpretation; do not invent residual diarization or identity matching.
9. **WER:** use identical, versioned Unicode/text normalization and word
   tokenization on reference and hypothesis, proposed as Unicode NFC, lowercase,
   punctuation removal, whitespace normalization, and whitespace tokenization.
   Recognize rather than translate. For automatic language selection, detect
   from the reference channel and use the same language for its separation.
   Allow an explicit language override. Whitespace WER has language limitations;
   document them and avoid quietly switching to CER or introducing language-
   specific segmenters. Preserve insertion counts for empty reference text;
   corpus WER is `null` when total reference words are zero, including all-empty
   evaluations. Positive-reference corpus WER includes those insertions and may
   exceed 1.0.
10. **Availability:** default `--metrics all` means the full protocol. Offer an
    explicit `core` selection for SI-SDRi, activity, STOI, and STOI improvement,
    plus individual metric selection. Missing requested packages/models cause a
    clear preflight error, rather than a silently reduced evaluation. Valid
    per-utterance ineligibility and missing residual channels remain reportable
    outcomes. A selected subset must be labeled as a partial protocol evaluation.

### Silence and short-speech behavior

Keep the whole input triple even when its reference T channel is silent or has
no qualifying utterances. Evaluate R independently. With no raw reference T
activity there are no target reference utterances or overlap regions; their
score lists are empty. Short raw speech discarded by utterance filtering still
contributes to activity and can qualify for the separate overlap rules.

Full-channel activity always runs. Speech on an inactive reference channel adds
false positives; missing reference speech adds false negatives. Correct silence
adds no TP/FP/FN, and undefined ratios remain `null` rather than an invented
perfect score. Record reference-inactive recording counts and active-output
frames on those recordings so their behavior remains visible in corpus reports.

When reference speech is present but its estimated slice is all zero, SI-SDRi is
undefined and retains a diagnostic. Inspection of TorchMetrics' stabilized
formula shows that accepting its output would instead yield 0 dB SI-SDR, possibly
giving a silent estimate a positive improvement over a negative mixture baseline.
[SI-SDR implementation](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/sdr.py).
STOI can return a valid zero for that estimate when enough reference frames
remain, giving a negative improvement relative to an intelligible mixture;
PESQ/ViSQOL outcomes follow their documented backend behavior. Do not discard
the recording or filter reference utterances by separated VAD.

DNSMOS uses whatever separated utterances exist, including false-positive
speech, and measures their quality rather than correctness of identity/timing.
Silence supplies no separated utterance for DNSMOS or similarity. ASR still runs
on full channels: empty reference transcripts retain insertion counts, and empty
hypotheses against nonempty references contribute deletions. Do not force ASR
text to empty using the evaluation VAD; record that ASR can itself hallucinate.

### VAD configuration rationale

Relative to Silero's current timestamp defaults, threshold 0.5, exit threshold
0.35, and minimum silence 100 ms are unchanged. The exit threshold provides
hysteresis: speech starts at 0.5, but ending it requires probabilities below 0.35
for the configured silence interval. The returned end boundary is near the
beginning of that silence, rather than including the entire waiting period.
The two changes are `speech_pad_ms=0` instead of 30 and
`min_speech_duration_ms=0` instead of 250. These preserve short detections and
avoid expanding activity into adjacent silence before protocol-specific interval
processing. The protocol then applies its own 0.5-second utterance and 0.2-second
overlap filters. Raw timestamp intervals still reflect Silero's window resolution
and minimum-silence behavior; they are not sample-perfect speech labels.
[Silero timestamp API](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py).

### Speaker model rationale

VoxBlink2 SimAMResNet34 is an efficient multilingual deployment starting point
with an official ONNX export. WeSpeaker
lists approximately 25.2M parameters for this model and 50.2M for SimAMResNet100;
the latter has better reported VoxCeleb verification error in its results table.
Make the model asset/frontend configuration explicit so SimAMResNet100 can be
used without changing metric code. Fix the chosen model across comparisons.
Verification benchmarks do not establish accuracy on these phone/meeting
channels or multi-speaker residuals.
[Pretrained models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md?plain=1),
[verification results](https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/README.md).

## Proposed implementation structure

Keep `eval_cli.py` responsible for argument parsing, logging, discovery, and
calling an evaluation pipeline. Put implementation under
`src/realtime_speech_separation/evaluation/`:

- `types.py`: small typed configuration, audio/interval, observation, and result
  structures; no model loading or optional metric imports.
- `audio.py`: triple loading and validation.
- `vad.py`: Silero adapter, interval operations, utterances, overlap, activity.
- `metrics.py`: focused SI-SDRi/STOI/PESQ/ViSQOL/DNSMOS adapters.
- `speaker.py` and `asr.py`: specialist model/backend adapters.
- `reporting.py`: schema serialization, compatible-result loading, aggregation,
  and CSV generation.
- `pipeline.py`: per-recording preparation, selected metrics, resource reuse,
  and cache coordination.

Use small functions and dependency injection for model/backend calls. A small
fixed dispatch table is sufficient; do not build a plugin framework. Import
optional backends only when requested. Load each neural model/session once per
process, use inference mode, default to CPU, and bound CPU threads. Keep CLI help
and package import usable without evaluation models or optional backends.

### Results and aggregation contract

Use SI-SDRi as the primary reconstruction score and absolute PESQ, STOI, and
ViSQOL as the primary intrusive perceptual scores. Activity, speaker cosine,
DNSMOS-P.808, and WER remain primary measurements in their respective categories.
Retain absolute SI-SDR, mixture baselines, and STOIi as supplementary diagnostics
in detailed results and clearly label them in summaries. Do not count the
absolute/improvement forms as independent evidence or automatically add PESQi,
ViSQOL improvement, or analogous variants for every metric.

This is a reporting choice supported by examples rather than a universal
literature rule: [Kealey et al.](https://www.isca-archive.org/interspeech_2024/kealey24_interspeech.pdf)
report SI-SDRi with absolute PESQ/STOI;
[Wu et al.](https://www1.se.cuhk.edu.hk/~hccl/publications/pub/C2AV-TSE_Context_and_Confidence-Aware_Audio_Visual_Target_Speaker_Extraction.pdf)
report both SI-SDR forms with absolute PESQ/STOI; and
[Li et al.](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-spr.2019.0290)
also report PESQ/STOI improvements.

Absolute scores describe the resulting output, while improvements describe its
change relative to the mixture. For an identical set of valid utterances and
weights, mean improvement equals mean output score minus mean mixture baseline;
subtracting that common baseline does not change separator rankings. Different
eligibility/failure populations can break that comparison. Preserve paired
per-utterance differences and their coverage; do not derive improvement
distributions by subtracting independently aggregated percentiles.

Define one result schema before adding metric adapters. Each per-separation result
should contain:

- Protocol/evaluator revisions, inference hash, evaluation configuration ID,
  resolved configuration, backend versions, and model/native-runtime identities.
- Relative mixture/reference/separated paths, source content identities, original
  rates/channel counts, evaluation rate/sample count, and available output channels.
- Per-channel raw VAD and utterance intervals, plus reference overlap intervals;
  intervals retain IDs, sample bounds, span, and original active duration.
- Activity TP/FP/FN and frame count for each available channel pair.
- WER substitutions/deletions/insertions/reference-word counts, transcription
  status, and raw/normalized full-channel transcripts or cache references.
- Per-interval observations for `si_sdri`, `overlap_si_sdri`, `stoi`, `stoii`, `pesq`,
  `visqol`, `dnsmos_p808`, and `speaker_cosine`, separately under T/R. Each contains
  its interval ID/origin, status, value or `null`, and an eligibility/failure reason.
  Improvement observations also retain their input/output metric values.
- Metric/channel coverage: candidate, eligible, attempted, scored, ineligible,
  undefined, and failed counts, with reason counts. Track silent estimates on
  reference utterances even when no separated utterance exists. Disabled metrics
  and unavailable channels have explicit statuses rather than misleading zeros.

Candidate populations must be stable and metric-specific: reference channel
utterances for intrusive scores, reference overlap intervals for overlap SI-SDRi,
and separated channel utterances for DNSMOS and speaker similarity. Report both
`scored / candidate` and `scored / eligible`; use `null` for an empty denominator.
For activity and WER report recording/frame/word coverage in their own units.
Always retain failures in the denominator appropriate to their population.

A **candidate** belongs to the metric's source population before metric-specific
eligibility checks. An **eligible** candidate meets those checks, such as the
3-second span requirement and an available reference prototype for similarity.
Successful execution is separate: an eligible utterance can still fail extraction
or produce an undefined value. For example, ten separated utterances can yield
seven duration-eligible candidates and six valid similarities: 60% coverage of
all candidates and approximately 85.7% of eligible candidates. Report reference
prototype eligibility/extraction separately, including short-reference exclusions.

Aggregate separately by inference hash **and evaluation configuration ID**:

- Sum activity counts before calculating P/R/F1; do not average recording ratios.
- Sum WER counts before calculating WER; do not average recording WERs.
- Flatten valid utterance observations across recordings, separately per metric,
  channel, and overall/overlap condition. Compute mean, min, p10, q1, median, q3,
  p90, and max; retain current population SD if useful. Use a fixed NumPy
  percentile method, recorded in configuration. Do not average recording means
  or weight utterances by duration.
- Include observation counts, pooled coverage, failure reasons, and processed,
  invalid, reused, target-only, and failed recording counts. A failure of one
  metric must not erase other scores from that recording.
- Empty distributions have count zero and `null` statistics. Reject or sanitize
  nonfinite observations before serialization; use strict JSON output.

The evaluation configuration ID hashes protocol/implementation revision,
selected metrics, parameters, backend versions, model digests, and native-runtime
identities. Source audio content hashes are separate per-result validity keys;
including them in the configuration ID would prevent corpus aggregation.

Use configuration-qualified output names such as
`<file_id>_results_<inference_hash>_<evaluation_id>.json`,
`summary_<inference_hash>_<evaluation_id>_<selection_id>.json`, and
`metrics_<inference_hash>_<evaluation_id>_<selection_id>.csv`.

`--input_filter` constrains both processing and aggregation, including cached
results: enumerate matching mixture triples, load/compute only those results,
and summarize that collection. Never scan arbitrary result JSONs for aggregation.
Keep filter selection out of the evaluation configuration ID so per-sample work
is reusable across dataset and overall reports. Use a separate `selection_id`
(`all` for no filter, otherwise a digest of normalized filter terms), and include
the filter terms and actual selected mixture paths in report provenance. This
lets dataset-specific and overall report files coexist.

Keep the primary CSV at one row per separation, with per-channel scalar summaries,
raw counts, statuses, and coverage. Add
`utterances_<inference_hash>_<evaluation_id>_<selection_id>.csv` with one row per observation,
including failed/ineligible observations. CSV recording means are conveniences;
the corpus summary must be computed from raw observations/counts.

## Implementation stages

Stages are ordered session-sized work packages. Each can be started in a fresh
session by reading this plan, the linked protocol, and the named files. Complete
its prerequisite stages first. Stage-local tests below are future work, not tests
to run during this planning task. Record decisions, completion status, validation,
and deviations here when implementing each stage.

### Stage 1 — Freeze contracts and introduce result/reporting foundations

**Prerequisites:** none. **Read:** protocol, current CLI, package tests, dependency
manifest, and the contracts above.

**Deliverables:**

- Resolve the proposed edge-case policies and record accepted metric semantics
  in `docs/evaluation-protocol.md`, linking implementation details to this plan.
  Put new CLI usage instructions in this plan or the README, preserving the
  protocol's existing detail.
- Add `evaluation/types.py` and the pure parts of `reporting.py`: result structure,
  typed interval/observation/count records, strict serialization, configuration
  identity, pooled aggregation, and coverage accounting.
- Choose an explicit evaluator revision marker to invalidate changed scoring
  semantics even while package version remains `0.1.0`.
- Establish a compatible package/version strategy for Python 3.12 and supported
  package Python versions. Keep dependency additions with their owning stages;
  do not install every backend to develop the reporting layer.
- Define the reporting contract independently of CLI integration, which is
  completed in Stage 4.

**Validation to implement:** synthetic result records showing unequal utterance
counts per recording, count-weighted WER/activity, metric-local failures,
target-only records, zero denominators, empty/singleton distributions, strict JSON
round trips, and invalid structure/configuration rejection. These tests need
only base/dev dependencies and no audio models.

**Completion boundary:** the schema and summarizer can represent every protocol
measurement and produce correct summaries from hand-constructed records.

### Stage 2 — Validate portable audio triples and harden discovery

**Prerequisites:** Stage 1. **Read:** CLI discovery helpers, `data_utils.py`,
`audio_utils.py`, and inference delay removal in `inference_cli.py`.

**Deliverables:**

- Add `evaluation/audio.py` with float32 16 kHz loading and validated shapes:
  mixture `(n,)`, reference `(2, n)`, separated `(1, n)` or `(2, n)`. Accept
  target-only input as `(n,)` as well, normalizing it to `(1, n)`.
- Remove evaluation's enrollment requirement when wiring the new path; retain
  enrollment behavior in inference and input extraction unchanged.
- Define structured per-triple errors for missing files, decode failures,
  unsupported channel counts, empty/nonfinite audio, and mismatched lengths.
- Match adjacent separated audio by exact mixture-stem prefix and supported
  suffix; ignore JSON sidecars/directories and process deterministic ordering.
  Extract the entire inference suffix rather than splitting arbitrary underscores.
- Preserve `--input_filter` semantics. Avoid global discovery changes that alter
  unrelated CLIs unless necessary. Temporary backend WAVs must live outside the
  scanned tree; evaluator cache directories must not become new inputs.

**Validation to implement:** temporary synthetic WAV triples at different native
rates, float and PCM encodings, stereo and mono estimates, missing enrollment,
wrong channel counts, mismatched lengths, sidecar matches, multiple inference
hashes, and filtered nested paths. No source dataset is needed.

**Completion boundary:** a validated triple is available to subsequent stages,
or a precise recording-level error is returned without loading any model.

### Stage 3 — Add self-contained Silero preprocessing and activity counts

**Prerequisites:** Stages 1–2. **Read:** protocol preprocessing/activity sections
and `scripts/data_prep/prep_vad.py` for API familiarity only. Do not import that
dataset preparation script or reuse its output files.

**Deliverables:**

- Add a lazy Silero adapter in `evaluation/vad.py`. Load its model once; reset
  state for every independent channel/recording. Request sample timestamps and
  apply the explicit VAD configuration.
- Run VAD separately on reference T/R and every available separated channel.
  Prepare reference information once per mixture and reuse across separation
  variants; do not reuse separated VAD across distinct outputs.
- Implement interval validation/union/intersection and protocol utterance
  construction: merge gaps up to 1 second, then require both span and original
  active duration of at least 0.5 seconds.
- Derive overlap from unmerged reference T/R activity, merge gaps, require
  0.2-second span and actual intersection activity, then pad to 0.5 seconds using
  the boundary policy. Never substitute the intersection of merged utterances.
- Build exactly `ceil(n / 1600)` activity frames at 16 kHz from raw channel VAD.
  Calculate TP/FP/FN independently for T/R using the majority policy.
- Preserve preprocessing intervals and diagnostics needed by result records.

**Validation to implement:** inject predetermined VAD intervals into CPU tests.
Cover 1-second gaps, exact threshold boundaries, long spans with too little
active speech, fragmented overlaps, merged-gap false overlaps, padding at both
ends, no speech, no overlap, exact-half frames, and final partial frames. Verify
calls on all available channels, state reset, and reference reuse using a fake
Silero adapter. Real-model tests remain opt-in.

**Completion boundary:** portable preprocessing returns all interval populations
and activity counts using only triple waveforms and the selected Silero model.

### Stage 4 — Integrate core metrics and the new CLI execution path

**Prerequisites:** Stages 1–3. **Read:** protocol SI-SDRi/STOI/aggregation sections
and the Stage 1 schema.

**Deliverables:**

- Add functional SI-SDR and standard STOI adapters in `evaluation/metrics.py`.
  For every reference utterance and overlap region, slice estimate/reference/
  mixture at identical bounds. Compute SI-SDRi for T/R separately, without PIT.
  Retain input/output SI-SDR and their difference.
- Run STOI on reference-defined utterances. Catch documented insufficient-frame
  warnings and metric-specific errors without accepting failure sentinels.
  Also compute mixture STOI on identical reference slices and report
  `stoii = STOI(reference, separated) - STOI(reference, mixture)` as a
  supplementary diagnostic. Reuse mixture
  baselines across separation variants. Positive values mean improvement; report
  raw differences rather than relative percentages. A missing baseline prevents
  improvement scoring without erasing a valid output STOI. Do not gate intrusive
  evaluation on separated VAD.
- Guard zero-energy reference/estimate/mixture slices before SI-SDR; preserve
  undefined observations and silent-estimate diagnostics. Valid zero STOI must
  remain a score, distinct from an insufficient-frame sentinel or backend error.
- Add `pipeline.py` and wire validated loading, VAD/activity, core metrics,
  per-separation results, and pooled reports into `eval_cli.py`.
- Add metric selection with fixed choices and resource preflight. `--metrics core`
  becomes runnable here; `all` must clearly reject not-yet-implemented backends
  until later stages supply them. Preserve input/filter/overwrite/logging flags;
  reject `--compute_permutation` and non-16-kHz evaluation with clear messages.
- Introduce configuration-qualified result names, selection-qualified report
  names, source identity checks, and atomic per-result writes. Cached results
  participate in summaries only when their mixtures match the invocation filter.
- Replace Asteroid usage and update `eval` dependencies to explicit Silero,
  TorchMetrics, and pystoi requirements. Keep unused backends lazily imported.

**Validation to implement:** fixed-seed signals where an estimate equal to the
mixture has zero improvement; degraded vs. improved estimates; fixed channel
identity on swapped outputs; interval-specific mixture baselines; silent estimate
and zero-energy reference handling; valid zero STOI and negative STOI improvement;
variable utterance counts; and a complete
core CLI run with synthetic audio and fake VAD. Test cached reruns and missing
requested backend errors without downloading models.

**Completion boundary:** the core subset works end to end on standalone triples
and generates protocol-shaped results with correct utterance/count weighting.

### Stage 5 — Add utterance-level PESQ

**Prerequisites:** Stage 4. **Read:** PESQ protocol section and adapter error contract.

**Deliverables:**

- Add the wideband 16 kHz PESQ adapter over each reference utterance, independently
  for T/R. Preserve sample alignment and pass NumPy arrays to the existing wrapper.
- Add PESQ to the optional quality extra and resource preflight; document native
  build requirements and compatible Python/NumPy versions found during setup.
- Map documented no-utterance/too-short errors to observation status/reason.
  Distinguish invalid input from backend/resource failure. A PESQ failure must
  leave SI-SDRi, STOI, and activity observations intact.

**Validation to implement:** fake-adapter selection/slicing/failure tests in the
default suite; optional small CPU tests against the installed PESQ wrapper for
valid signals and documented error cases. Never count `-1` as a valid score.

**Completion boundary:** selecting PESQ produces per-reference-utterance scores
and honest coverage without adding native dependencies to base/dev installs.

### Stage 6 — Add official speech-mode ViSQOL

**Prerequisites:** Stage 4; independent of Stage 5. **Read:** official ViSQOL API
and protocol intrusive-quality section.

**Deliverables:**

- Initialize the official API once with 16 kHz speech scoring and the matching
  speech model asset. Return MOS-LQO for each reference utterance/channel.
- Expose a model path where needed and record native library/model identity.
  Keep native package provisioning explicit and out of runtime auto-installation.
- Determine the pinned backend's actual minimum usable patch/speech requirements.
  The upstream 8–10-second input guidance is a recommendation, not a reason to
  silently replace the protocol's utterances with arbitrary long windows.
  Report unsupported short utterances as ineligible/failed with a reason.
  Do not add metric-specific waveform padding or concatenate unrelated utterances
  without documenting an approved protocol change.
- Preflight a missing API/model with actionable setup information. If the native
  Python API cannot be supported on the target environment, document that finding
  and use an adapter to the official CLI, preserving the same scoring mode and
  structured output; do not create a new ViSQOL implementation.

**Validation to implement:** mocked configuration and per-utterance calls, model
absence, very short speech, and native failure isolation. Add an opt-in local
backend smoke test to establish short-utterance behavior and official API parity.

**Completion boundary:** ViSQOL runs through an official backend with recorded
configuration and documented eligibility, without changing the other metrics.

### Stage 7 — Add DNSMOS-P.808 on separated utterances

**Prerequisites:** Stage 4; independent of Stages 5–6. **Read:** protocol DNSMOS
section and the selected TorchMetrics DNSMOS implementation.

**Deliverables:**

- Add a lazy CPU DNSMOS adapter with cached sessions and bounded threads. Feed
  each separated-defined utterance, select output component `p808_mos`, and
  retain one score per utterance. Do not average P.808 with P.835 outputs or
  apply personalized/P.835 calibration to P.808.
- Retain the library's short-input extension and internal-window averaging;
  document these as backend behavior rather than adding a 9-second utterance
  eligibility cutoff. Guard empty input before calling it.
- Add CPU ONNX Runtime/requests to the quality extra. Document every asset that
  the selected public API requires, including any companion model loaded even
  though only P.808 is reported. Pin/checksum the assets.
- Support pre-provisioned offline execution. Current upstream uses its own
  `~/.torchmetrics/DNSMOS` cache and has no public model-path argument; do not
  promise a general cache flag that it ignores. Preflight that cache and document
  provisioning. If a configurable model location is required, assess a pinned
  speechonnxmetrics P.808 adapter before using private TorchMetrics internals or
  writing new DNSMOS signal processing.

**Validation to implement:** a fake four-component return verifies P.808 selection;
tests verify separated rather than reference boundaries, empty input handling,
session reuse, asset failures, and metric-local coverage. An opt-in real backend
test covers both short and long utterances with pre-provisioned weights.

**Completion boundary:** P.808 scores are correctly identified, reproducible,
and independent of the reference utterance population.

### Stage 8 — Add multilingual WeSpeaker similarity

**Prerequisites:** Stage 4; independent of Stages 5–7. **Read:** speaker protocol,
official ONNX inference frontend, and pretrained model documentation.

**Deliverables:**

- Add `evaluation/speaker.py` around the official multilingual VoxBlink2
  SimAMResNet34 ONNX asset, with explicit model configuration supporting the
  corresponding SimAMResNet100 export. Use the model's published filterbank frontend,
  amplitude scaling, frame settings, and mean normalization. Reuse library
  filterbank routines; do not derive a new feature extractor.
- Add only deployment dependencies to `eval-speaker`: CPU ONNX Runtime and the
  matching torchaudio frontend, sharing Torch already needed for evaluation.
  Require a local model path and record its digest and frontend settings.
- Validate adapter embeddings against the official WeSpeaker extractor using
  the identical model/frontend before accepting the lightweight deployment path.
  Preserve attribution if adapting the upstream inference example.
- Extract reference utterance embeddings only when they contain at least
  3 seconds of VAD-active speech and form one equally averaged prototype per
  recording/channel. Report excluded short references; do not fall back to
  unstable embeddings when none qualifies. Cache reference artifacts across
  inference variants. Score separated utterance embeddings only when their
  utterance span is at least 3 seconds.
- Compute raw cosine in `[-1, 1]`. WeSpeaker's convenience similarity method maps
  cosine to `[0, 1]`; do not use that transformed value for this metric.
  [Convenience method implementation](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/cli/speaker.py).
- Report short utterances, missing/invalid prototypes, zero-norm embeddings, and
  extraction failures explicitly. Do not use enrollment audio, language-specific
  identity assumptions, diarization, or channel reassignment.

**Validation to implement:** deterministic fake embeddings test prototype scope,
equal weighting, exclusion of short reference utterances, merged silent gaps,
no eligible prototype, the exact 3-second boundary, negative cosine, missing
residuals, and extraction failures. Keep
model/frontend parity and real multilingual model runs opt-in.

**Completion boundary:** similarity follows the protocol's asymmetric reference
prototype/separated-utterance rules and reproduces official model inference.

### Stage 9 — Add full-channel whisper.cpp transcription and count-based WER

**Prerequisites:** Stage 4; independent of Stages 5–8. **Read:** protocol
transcription/WER sections, pywhispercpp's model API, and JiWER usage.

**Deliverables:**

- Add `evaluation/asr.py` around an in-process pywhispercpp model loaded once
  from a local large-v3-turbo asset. Pin the binding/native build, CPU/thread
  settings, decoding parameters, and quantization if explicitly changed. Disable
  automatic GPU selection and model downloads during evaluation.
- Transcribe complete reference T/R and available separated channels. Do not
  transcribe the mixture, split ASR by evaluation utterance, or mask non-speech
  away before transcription. Cache reference transcripts across inference variants.
- Pass contiguous mono float32 arrays at 16 kHz directly to `Model.transcribe`
  and concatenate returned segments in time order. No temporary WAV, PCM16
  conversion, FFmpeg decoding, or external `whisper-cli` process is required.
- Use deterministic settings where available, transcription rather than
  translation, and agreed language selection. Explicitly reset decode parameters
  and disable context/prompt carryover between independent channels/files; the
  binding can retain per-call parameter overrides. Record detected language when
  available and validate the pinned binding's language-detection API.
- Handle model-load, inference, malformed-return, and cancellation errors.
  Use a cooperative abort callback for time limits if supported by the pinned
  release; do not promise a subprocess-style hard timeout for in-process native
  code. Document any binding limitation rather than adding an external CLI fallback.
- Add pywhispercpp and JiWER 4+ in `eval-asr`, apply shared normalization, and retain
  S/D/I and reference words (`hits + substitutions + deletions`). Cache both
  raw and normalized transcripts with recognizer/normalizer identities.
- Retain empty-reference insertion counts and apply the Stage 1 denominator
  policy. Label this as agreement with reference ASR, since no human transcript
  is supplied. Residual multi-speaker transcription remains one channel stream.

**Validation to implement:** fake binding/model outputs and deterministic text pairs
cover edit counts, unequal reference lengths, empty references/hypotheses,
Unicode/punctuation normalization, language consistency, cancellation, malformed
segments, model reuse, and no prompt/parameter leakage between channels.
A requested WER run must fail preflight when its model/binding is absent.
Real large-v3-turbo transcription remains an opt-in resource test.

**Completion boundary:** full-channel WER counts and reproducible transcripts are
available without any dataset transcripts or remote ASR service.

### Stage 10 — Finish caching, reports, documentation, and full-protocol validation

**Prerequisites:** Stages 1–9. **Read:** all result producers, the current README
evaluation section, architecture evaluation section, and evaluation output
consumers (including shell scripts and the run-eval skill documentation).

**Deliverables:**

- Complete shared reference preprocessing caches keyed by source digest,
  frontend settings, model identity, and relevant preprocessing parameters.
  Cache separated preprocessing by its own source digest. A simple in-process
  cache is sufficient initially; add persistent reference artifacts only where
  they materially avoid VAD, embedding, or ASR work across reruns.
- Finalize compatible-result skipping and `--overwrite`: matching successful
  results are reusable; stale/incompatible results are recomputed or
  clearly rejected. Do not treat interrupted/failed evaluations as permanently
  completed. A short/ineligible or mathematically undefined observation is
  distinct from a transient backend failure. Write caches/results atomically
  and rebuild summaries from results in the current filtered invocation only.
- Keep source fingerprints separate from evaluation identity, prevent duplicate
  result inclusion, and isolate configurations in filenames and summaries.
  A native-runtime/model/settings change must invalidate the affected work.
- Finish recording and utterance CSVs, pooled distributions, count-derived ratios,
  per-metric coverage/reasons, unavailable-channel reporting, and clear partial-
  versus full-protocol labeling. Preserve complete records for other metrics
  when a single backend fails. Use a nonzero exit status for configuration,
  invalid-input, or requested-backend execution failures; valid ineligibility
  or mathematically undefined scores alone are not a failed evaluation.
- Handle no discovered mixtures/separations and all-invalid/all-ineligible
  evaluations deliberately, with useful diagnostics and zero-count reports where
  an inference configuration is known. Eliminate first-result indexing assumptions.
- Ensure `--metrics all` now executes every requested protocol family; list
  explicit setup requirements and fail early for missing resources. CPU must
  remain a supported default even if full ASR is slow. Do not automatically
  select a shared GPU or silently substitute smaller/different models.
- Document portable directory examples, install extras/native assets, offline
  provisioning, model/cache locations, metric subsets, output naming/schema,
  coverage, silence diagnostics, filtered reports, and costs. Keep README commands
  environment-manager neutral. Update architecture links, including its existing misspelled
  `evaluation_protocol.md` link. Update existing output-layout documentation as
  needed without invoking the run-eval workflow during development.

**Validation to implement:** an offline end-to-end CLI test with synthetic WAVs
and fake backends for every metric. Include multiple nested mixtures, multiple
inference hashes, unequal utterance counts, target-only output, no speech/overlap,
one backend failure, an entirely empty metric distribution, stale caches,
changed audio/configuration/model identities, subset/full runs, input filtering,
overwrite, and repeated summary generation. Include dataset-filtered summaries
after an overall run with cached results: each dataset summary must contain
exactly its selected samples, and report files must coexist. Verify valid
JSON/CSV and pooled results from known observations. Preserve package tests.

**Completion boundary:** all protocol measurements, preprocessing, coverage, and
aggregation work on standalone triples with reproducible resources.

## Validation policy for implementation sessions

No checks or tests belong to this planning task. During later implementation,
add focused pytest tests with each substantive stage and run the repository's
required checks in the `audio` environment:

```bash
conda run --no-capture-output -n audio python -m ruff check .
conda run --no-capture-output -n audio python -m ruff format --check .
conda run --no-capture-output -n audio python -m pytest
```

Keep the common test suite offline and CPU-only, with injected model/backend
adapters and synthetic temporary audio. Keep real model/native backend validation
explicitly opt-in; document checks that cannot run because a resource is absent.
Do not make base/dev CI install the evaluation runtime stacks.

For a final real-data smoke run, copy a small set of existing triples to a
temporary location and use `--input_filter` and `--overwrite` there. Verify the
CLI works without access to the project's `data` tree or enrollment WAVs. Do not
modify source data or run separator inference merely to test evaluation. If any
explicitly requested validation uses a GPU, first inspect `nvidia-smi`, select the
least-used suitable GPU, and restrict it with `CUDA_VISIBLE_DEVICES` according to
the repository instructions.

## Review points before implementation

The main semantic choices for review are the raw Silero settings, reference
prototype scope/averaging, empty-denominator conventions, multilingual WER
normalization, and the default full-versus-explicit-core experience. Stage 1
records these choices so fresh sessions share one contract. The principal
integration uncertainties are native PESQ/ViSQOL support, lightweight WeSpeaker
frontend parity, and DNSMOS's fixed upstream cache location; their owning stages
have explicit validation and fallback boundaries. None requires access to
project-specific VAD or source-audio metadata.
