# Evaluation Protocol

The enrollment-conditioned speech separation task is known in the literature as 
Target Speaker Extraction (TSE).

Every sample to evaluate will consist of three audio recordings:
- **<file_id>.wav:** mono input mixture with multiple speakers. **Shape: (n,)**
- **<file_id>_reference.wav:** stereo reference recording where the target speaker (T)
  is isolated on the left channel and the residual speakers (R) 
  is on the right channel. **Shape: (2, n)**
- **<file_id>_separated\_<args_hash>.wav:** stereo output of the separator model under a particular
  set of inference arguments, an estimate of <file_id>_reference.wav. **Shape: (2, n)**.
  Target-only output may instead have **shape: (n,)** or **(1, n)**; evaluation
  standardizes both to **(1, n)**.

where:
- **n:** number of samples in the input mixture, reference, and separated recordings (must be the same across all three).

All three recordings must be temporally aligned and have equal sample counts
of at least 16,000 samples (one second) after resampling to 16 kHz.

Evaluation includes the following metrics:
1. **Separation Quality** - SI-SDRi, Speech Activity P/R/F1
2. **Speaker Similarity** - Cosine similarity of speaker embeddings
3. **Audio Quality & Perceptibility** - PESQ, STOI, ViSQOL, DNSMOS-P.808, WER

SI-SDRi is the primary reconstruction measure. PESQ, STOI, and ViSQOL are reported
as absolute output scores. Absolute SI-SDR, mixture baselines, and STOI improvement
are supplementary measurements for interpreting the primary results; each
absolute/improvement pair describes the same underlying metric.

Measurements are done independently on both the target (T) and residual (R) channels,
assuming the R channel is available in the separator model output. Some baseline separators
only generate the target (T) channel, in which case the residual (R) channel metrics cannot be computed.

## Preprocessing

#### Resampling

All recordings (input mixture, reference, separated) are loaded as float32 and resampled to 16 kHz:

```python
sr = 16000
input_audio, _ = librosa.load(input_file, sr=sr, mono=True, dtype=np.float32)
reference_audio, _ = librosa.load(ref_file, sr=sr, mono=False, dtype=np.float32)
separated_audio, _ = librosa.load(sep_file, sr=sr, mono=False, dtype=np.float32)
```

We expect input_file, ref_file, and sep_file to be stored as 32-bit floating point WAV (subtype="FLOAT") prior to evaluation,
but do not enforce this in the evaluation code itself.

#### Utterance Identification

1. [Silero VAD](https://github.com/snakers4/silero-vad) is used at 16 kHz to identify VAD-active regions
   from the T and R channels of the reference and separated recordings.
   Run VAD directly on the supplied WAVs, independently of any dataset VAD files.
   Use speech threshold 0.5, exit threshold 0.35, minimum silence 100 ms,
   timestamp padding 0 ms, and minimum speech duration 0 ms. Preserve these raw
   timestamp intervals before applying the following utterance filters.
2. Contiguous regions within 1 second are merged to form longer active regions. 
3. Regions shorter than 0.5 seconds or containing less than 0.5 seconds of active speech are discarded to ensure reliable evaluation.
4. Utterances are defined as the regions that remain after merging contiguous regions and discarding short regions.

Intervals use half-open sample bounds `[start, end)` at 16 kHz. A gap of exactly
1 second is merged; duration minima are inclusive. Keep the original active
sample count through merging, excluding silent gaps.

#### Overlap Identification

Overlap regions are identified by the temporal intersections of VAD-active regions on the T and R channels.

1. Start with the raw VAD-active regions on the T and R channels before merging or discarding short regions.
2. Compute the temporal intersections of these regions to identify overlap regions.
3. Contiguous overlap regions within 1 second are merged to form longer overlap regions.
4. Overlap regions shorter than 0.2 seconds or containing less than 0.2 seconds of active speech are discarded,
   and remaining overlap regions shorter than 0.5 seconds are padded evenly on both sides to reach 0.5 seconds.

At a recording boundary, move unavailable padding to the other side. For an odd
padding sample count, put the extra sample on the right before adjusting for
boundaries. Padding does not increase the original active duration.

#### Transcription

[whisper.cpp](https://github.com/ggml-org/whisper.cpp), through the in-process
[pywhispercpp binding](https://github.com/absadiki/pywhispercpp), is used to transcribe
the full T and R channels of the reference and separated recordings using the
whisper-large-v3-turbo model. Feed the full 16 kHz channel waveforms without
utterance segmentation or evaluation-VAD masking, and do not carry transcription
context between independent channels or recordings.

Use transcription, not translation. Detect the language independently from each
full reference channel and use that language for its corresponding separated
channel. An explicit language override applies to both reference and separated
channels. Record the resolved language and decoding settings.

#### Speaker Embeddings

[WeSpeaker toolkit](https://github.com/wenet-e2e/wespeaker) is used to extract speaker
embeddings using a model appropriate for multilingual speech. Both reference and
separated utterances must contain at least 3 seconds of raw VAD-active speech to
provide enough speech for estimating speaker identity. Silent gaps inside a
merged utterance do not count toward this minimum. Embed the full utterance span
once the detected speech meets the threshold.
The eligible reference embeddings on T and R are separately averaged with equal
utterance weight **within each evaluation recording** to create target and residual
speaker prototypes. If none qualifies on a channel, its prototype and speaker
similarity measurements are unavailable. Do not fall back to shorter reference
utterances or pool unrelated speakers across recordings.

Average the raw embeddings before cosine normalization; do not normalize each
utterance embedding before averaging. A failed required embedding extraction
makes that channel's prototype unavailable rather than dropping the utterance
and constructing a different prototype. A zero-norm prototype is also unavailable.

Extract one embedding per eligible separated utterance on T and R. Keep the
selected model and frontend fixed across separator comparisons.

#### Silent Channels and Missing Speech

A silent reference channel does not invalidate the evaluation sample. Its
reference-utterance metrics have no candidates; the other channel is evaluated
independently. Activity measurements still use raw VAD over the full recording,
so spurious output speech contributes false positives and missing reference speech
contributes false negatives. Raw speech too short to become an utterance still
contributes to activity and may qualify under the separate overlap rules.

When reference speech has an all-zero estimate, retain the reference utterance
and evaluate each metric according to its definition. SI-SDR and its improvement
are undefined for zero-energy estimates after centering; report `null` with an
explicit reason and a `separated_missing_utterances` diagnostic. Centering means
subtracting each interval waveform's arithmetic mean. These missing-utterance
diagnostics refer to zero-energy separated waveforms after that subtraction,
independently of whether separated VAD detects speech. Do not report a stabilized
numerical placeholder as a valid score. A zero-energy reference is ineligible;
a zero-energy mixture baseline makes improvement unavailable. A quiet nonzero
estimate must not be excluded merely because VAD detects no speech.

DNSMOS and similarity have no separated-utterance candidates when the output has
no qualifying speech. Full-channel transcription still runs. WER retains insertions
for empty reference transcripts and deletions for missing words; ASR can itself
hallucinate, so VAD must not silently replace its text with an empty transcript.

## Separation Quality

### SI-SDRi

Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) is a measure of how much of the
separator's output is explained by the reference vs. distortion or noise introduced
during the separation process.

SI-SDR Improvement (SI-SDRi) looks at `SI-SDR(separated, reference) - SI-SDR(input_mixture, reference)`
to see how much closer the separator output is to reconstructing the reference than just using the 
original input mixture.

SI-SDRi can be computed with or without permutation matching (i.e. taking the best score over all 
possible assignments of separated channels to reference channels). Since we have a target enrollment,
we expect the separated T channel to always correspond to the reference T channel and thus do not allow 
for permutation matching.

Use TorchMetrics' functional SI-SDR with `zero_mean=True` and float32 tensors
for both calls. Its [metric-backend implementation](https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/sdr.py)
uses `torch.finfo(preds.dtype).eps` (2**-23 for float32) in the projection numerator
and denominator and in both energy sums before the log ratio. Record the metric
backend version and these settings. Check centered zero energy before the metric-backend call,
as specified under silent channels; epsilon stabilization does not make those
undefined cases valid. Apply no positive energy cutoff to quiet nonzero signals.

We make the following SI-SDRi measurements:

#### Overall separation performance

1. Target SI-SDRi: Separated vs. Reference T channel SI-SDRi for each reference T channel utterance.
2. Residual SI-SDRi: Separated vs. Reference R channel SI-SDRi for each reference R channel utterance.

#### Sensitivity to simultaneous speech

1. Overlap-conditioned target SI-SDRi: Separated vs. Reference T channel SI-SDRi at reference Overlap regions.
2. Overlap-conditioned residual SI-SDRi: Separated vs. Reference R channel SI-SDRi at reference Overlap regions.

### Speech Activity Precision, Recall, and F1

Speech activity precision, recall, and F1 measure how well the separator's output matches the reference in terms of speech timing.
For example, the target should be active in the separator output if-and-only-if it is active in the reference target channel at that time.

For this evaluation, we break the full reference and separated recordings into 100ms frames:
```python
num_frames = math.ceil(reference.shape[-1] / int(0.1 * sr))
```
For all Separated and Reference channels, we construct a boolean mask indicating speech activity in each 100ms frame.
A frame is considered active if the majority of its span overlaps with VAD activity on the same channel.
The majority is strict: more than 50% is active; an exact tie is inactive.
Use the actual span of the last partial frame, without padding it to 100 ms.

*Note*: We use the raw VAD output, not the merged and filtered utterance intervals, to determine frame-level speech activity.

We then measure:
1. Target activity P/R/F1: Precision, recall, and F1 score for Separated vs. Reference T channel activity masks.
2. Residual activity P/R/F1: Precision, recall, and F1 score for Separated vs. Reference R channel activity masks.

## Speaker Similarity

### Speaker Embedding Cosine Similarity

Speaker embedding similarity measures how well the separated channels preserve the identities of the reference speakers.
Report the raw cosine similarity in `[-1, 1]`.

We measure:

1. Target speaker similarity: Cosine similarity between the Separated T channel utterances and the Reference T channel prototype.
2. Residual speaker similarity: Cosine similarity between the Separated R channel utterances and the Reference R channel prototype.

## Audio Quality & Perceptibility

### PESQ, STOI, ViSQOL

Perceptual Evaluation of Speech Quality (PESQ) measures the perceived audio quality of the separated channels compared to the reference channels.
Short-Time Objective Intelligibility (STOI) measures the intelligibility of the separated channels compared to the reference channels.
Virtual Speech Quality Objective Listener (ViSQOL) measures the perceived audio quality of the separated channels compared to the reference channels.

Use wideband PESQ at 16 kHz and report its MOS-LQO score. Record short buffers
and reference utterances where PESQ detects no usable speech as ineligible
intervals, with a reason for each exclusion.

Use ViSQOL in 16 kHz speech mode with the speech lattice model and report its
MOS-LQO score. Require at least three seconds of detected speech in the reference
utterance, excluding merged gaps.
For eligible spans, take at most the first nine seconds at identical bounds in
the reference and separated signals. Add 0.5 seconds of silence to each end of
both signals, yielding inputs of four to ten seconds. Report candidates at their
original reference utterance bounds.

We measure:

1. Target PESQ, STOI, ViSQOL: Scores between the Separated and Reference T channel for each reference T channel utterance.
2. Residual PESQ, STOI, ViSQOL: Scores between the Separated and Reference R channel for each reference R channel utterance.

### Supplementary STOI Improvement

For supplementary analysis, measure STOI improvement relative to the input
mixture on each reference channel utterance:

`STOIi = STOI(reference, separated) - STOI(reference, input_mixture)`

Both calls use the same reference waveform and time interval, with the reference
as the first argument. Positive values indicate improved intelligibility. Report
the absolute difference in STOI units; multiplying by 100 expresses percentage
points, not a relative percentage change. Retain both the mixture and separated
scores, and average the per-utterance differences during aggregation. A missing
baseline makes improvement unavailable without erasing a valid separated STOI.
This reporting convention is used in
[speech separation research](https://www.isca-archive.org/interspeech_2018/tan18b_interspeech.pdf).

### DNSMOS-P.808

DNSMOS measures the perceptual quality of the separated channels using a deep neural network-based model.

Report the non-personalized P.808 MOS at 16 kHz. Require at least three seconds
of detected speech in the separated utterance, excluding merged gaps. Pass each
eligible span in full to TorchMetrics, which repeats audio shorter than its
analysis window and averages internal window scores into one score per utterance.

We measure:

1. Target DNSMOS: DNSMOS score for each Separated T channel utterance.
2. Residual DNSMOS: DNSMOS score for each Separated R channel utterance.

### WER

Word Error Rate (WER) measures the transcription accuracy of the separated channels compared to the reference channels.

We measure:

1. Target WER: WER between the separated and target transcriptions of the T channel.
2. Residual WER: WER between the separated and residual transcriptions of the R channel.

Apply the same normalization to both transcripts in this order: Unicode NFC,
Unicode lowercase, removal of characters in Unicode punctuation categories
(`P*`), and whitespace normalization using split/join. Remove punctuation rather
than replacing it with spaces (for example, `can't` becomes `cant`); retain other
symbols. Tokenize on whitespace. Record raw and normalized text and the
normalizer identity with the recognizer configuration.

These measurements express agreement with reference ASR, rather than human
transcription accuracy. Whitespace WER has limited usefulness for languages whose
words are not normally separated by spaces, including Chinese and Japanese; do
not silently introduce language-specific tokenization between comparisons.

## Aggregation

- Activity P/R/F1 can be aggregated across the evaluation set by summing up the true positives, false positives, and false negatives across all recordings and then calculating precision, recall, and F1 score from these aggregated counts.
- WER can be aggregated across the evaluation set by summing up the substitutions, deletions, insertions, and reference word counts across all recordings and then calculating the overall WER from these aggregated counts.
- All other metrics can be aggregated across the evaluation set by averaging the per-utterance scores, regardless of the recording they belong to.

Activity precision, recall, or F1 is `null` when its own denominator is zero.
Corpus WER is `null` when total reference words are zero; otherwise include all
insertions, including those from empty reference transcripts. A recording with
zero reference words therefore still contributes insertions to a corpus with
nonzero reference words. Never average recording-level WER or activity ratios.

For each evaluation sample (consisting of the input mixture, reference, separated recordings),
the evaluation CLI should output a JSON file with:
1. Activity TP, FP, FN counts.
2. Substitutions, deletions, insertions, and reference word counts.
3. List of per-utterance scores for all other metric measurements.

When the evaluation CLI generates the summary report JSON, it should:
1. Sum up all Activity TP, FP, FN counts across all evaluation samples and calculate the aggregated precision, recall, and F1 score.
2. Sum up all substitutions, deletions, insertions, and reference word counts across all evaluation samples and calculate the aggregated WER from these counts.
3. Average the per-utterance scores for all other metrics across all evaluation samples to obtain the aggregated scores.
   In addition to the mean, also compute the distribution (min, p10, q1, median, q3, p90, max).

Use NumPy's `linear` percentile method and population standard deviation
(`ddof=0`). An empty distribution has count zero and `null` statistics; a
singleton has identical quantiles and zero standard deviation. Improvement
distributions use the paired per-interval differences, retaining available
absolute output scores even when the mixture baseline is unavailable.

## Reporting

Support separate reports for the full validation or test set and for selections
by source dataset. Dataset reports are primary evaluation results: two-speaker
recordings have one residual speaker, while multiparty recordings can have two
or more. Residual similarity to a single averaged prototype can therefore be
lower on multiparty audio, even with good separation. Each report must summarize
only its selected recordings and coexist with reports for other selections.

Not all utterances will qualify for all metrics. A metric's **candidates** are
the utterances in its source population before its specific eligibility rules;
**eligible** candidates meet those rules, and **scored** candidates successfully
produce a defined value. Report scored/candidate and scored/eligible coverage,
with counts and reasons for ineligible, undefined, and failed interval results.
Report reference prototype eligibility separately from separated speaker-score
coverage. Undefined ratios and empty score distributions are `null`.

Every candidate has one outcome: `scored`, `ineligible`, `undefined`, or `failed`.
Undefined and failed interval results count as eligible; only `ineligible` removes
a candidate from the eligibility denominator. Run eligibility checks before
speaker-backend extraction or metric-backend scoring. Disabled metrics and missing
output channels have explicit availability statuses and contribute no interval candidates.

Preserve recording-level results when one channel or metric is unavailable.
Include counts of inactive reference channels and reference utterances with silent
separated output so the valid-score means do not conceal missing or spurious speech.



Report recording participation separately for every channel measurement, in
addition to interval coverage. Count recordings with at least one scored value,
recordings with no candidates after successful preprocessing, recordings fully
excluded due to errors, and recordings with partial interval failures. A full
error exclusion means required preprocessing failed or every interval candidate
failed. Undefined and ineligible outcomes are not errors; partial failures retain
all successful scores. Disabled measurements and missing output channels do not
enter these recording candidate counts. Invalid audio is counted separately in
the recording totals because its channel populations are unknown.

Include exclusion counts by reason, counting each distinct reason at most once
per excluded recording. Multiple reasons can apply to one recording, so their
sum can exceed the number of excluded recordings. A reference-channel VAD
failure excludes that channel's reference-utterance measurements and both
channels' overlap measurements. Activity requires both reference and separated
VAD for its channel; separated VAD failures do not exclude reference-utterance
measurements. Keep these failed prerequisites distinct from successful VAD
finding no qualifying speech.
