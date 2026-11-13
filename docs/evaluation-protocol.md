# Evaluation Protocol

The enrollment-conditioned speech separation task is known in the literature as 
Target Speaker Extraction (TSE).

Every sample to evaluate will consist of three audio recordings:
- **<file_id>.wav:** mono input mixture with multiple speakers. **Shape: (n,)**
- **<file_id>_reference.wav:** stereo reference recording where the target speaker (T)
  is isolated on the left channel and the residual speakers (R) 
  is on the right channel. **Shape: (2, n)**
- **<file_id>_separated.wav:** stereo output of the separator model, an estimate
  of <file_id>_reference.wav. **Shape: (2, n)**

where:
- **n:** number of samples in the input mixture, reference, and separated recordings (must be the same across all three).

For evaluation, all three audio recordings must be at the same sampling rate and temporally aligned.

Evaluation includes the following metrics:
1. **Separation Quality** - SI-SDRi, Speech Activity P/R/F1
2. **Speaker Similarity** - Cosine similarity of speaker embeddings
3. **Audio Quality & Perceptibility** - PESQ, STOI, ViSQOL, DNSMOS-P.808, WER

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

We expect all audio files to be stored as 32-bit floating point WAV (subtype="FLOAT") prior to evaluation.

#### Utterance Identification

1. [Silero VAD](https://github.com/snakers4/silero-vad) is used at 16 kHz to identify VAD-active regions
   from the T and R channels of the reference and separated recordings.
2. Contiguous regions within 1 second are merged to form longer active regions. 
3. Regions shorter than 0.5 seconds or containing less than 0.5 seconds of active speech are discarded to ensure reliable evaluation.
4. Utterances are defined as the regions that remain after merging contiguous regions and discarding short regions.

#### Overlap Identification

Overlap regions are identified by the temporal intersections of VAD-active regions on the T and R channels.

1. Start with the raw VAD-active regions on the T and R channels before merging or discarding short regions.
2. Compute the temporal intersections of these regions to identify overlap regions.
3. Contiguous overlap regions within 1 second are merged to form longer overlap regions.
4. Overlap regions shorter than 0.2 seconds or containing less than 0.2 seconds of active speech are discarded,
   and remaining overlap regions shorter than 0.5 seconds are padded evenly on both sides to reach 0.5 seconds.

#### Transcription

[whisper.cpp](https://github.com/ggml-org/whisper.cpp) is used to transcribe the full T and R channels of the reference and separated recordings,
using the whisper-large-v3-turbo model.

#### Speaker Embeddings

[WeSpeaker toolkit](https://github.com/wenet-e2e/wespeaker) is used to extract speaker embeddings for all utterances on all channels in all recordings,
using a model appropriate for multilingual speech.
The reference embeddings on the T and R channels are separately averaged **across all** utterances to create target and residual speaker prototypes.
The separated embeddings on the T and R channels are taken for each utterance of at least 3 seconds to create utterance-level speaker embeddings.

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
num_frames = math.ceil(reference.shape[-1] / int(0.1*sr))
```
For all Separated and Reference channels, we construct a boolean mask indicating speech activity in each 100ms frame.
A frame is considered active if the majority of its span overlaps with VAD activity on the same channel.

*Note*: We use the raw VAD output, not the merged and filtered utterance intervals, to determine frame-level speech activity.

We then measure:
1. Target activity P/R/F1: Precision, recall, and F1 score for Separated vs. Reference T channel activity masks.
2. Residual activity P/R/F1: Precision, recall, and F1 score for Separated vs. Reference R channel activity masks.

## Speaker Similarity

### Speaker Embedding Cosine Similarity

Speaker embedding similarity measures how well the separated channels preserve the identities of the reference speakers.

We measure:

1. Target speaker similarity: Cosine similarity between the Separated T channel utterances and the Reference T channel prototype.
2. Residual speaker similarity: Cosine similarity between the Separated R channel utterances and the Reference R channel prototype.

## Audio Quality & Perceptibility

### PESQ, STOI, ViSQOL

Perceptual Evaluation of Speech Quality (PESQ) measures the perceived audio quality of the separated channels compared to the reference channels.
Short-Time Objective Intelligibility (STOI) measures the intelligibility of the separated channels compared to the reference channels.
Virtual Speech Quality Objective Listener (ViSQOL) measures the perceived audio quality of the separated channels compared to the reference channels.

We measure:

1. Target PESQ, STOI, ViSQOL: Scores between the Separated and Reference T channel for each reference T channel utterance.
2. Residual PESQ, STOI, ViSQOL: Scores between the Separated and Reference R channel for each reference R channel utterance.

### DNSMOS-P.808

DNSMOS measures the perceptual quality of the separated channels using a deep neural network-based model.

We measure:

1. Target DNSMOS: DNSMOS score for each Separated T channel utterance.
2. Residual DNSMOS: DNSMOS score for each Separated R channel utterance.

### WER

Word Error Rate (WER) measures the transcription accuracy of the separated channels compared to the reference channels.

We measure:

1. Target WER: WER between the separated and target transcriptions of the T channel.
2. Residual WER: WER between the separated and residual transcriptions of the R channel.

## Aggregation

- Activity P/R/F1 can be aggregated across the evaluation set by summing up the true positives, false positives, and false negatives across all recordings and then calculating precision, recall, and F1 score from these aggregated counts.
- WER can be aggregated across the evaluation set by summing up the substitutions, deletions, insertions, and reference word counts across all recordings and then calculating the overall WER from these aggregated counts.
- All other metrics can be aggregated across the evaluation set by averaging the per-utterance scores, regardless of the recording they belong to.

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

## Reporting
Not all utterances will qualify for all metrics. For example, speaker similarity is only measured for utterances of at least 3 seconds, and other metrics may have their own eligibility criteria. The reporting should clearly indicate the proportion of utterances considered for each metric, along with the aggregated scores.