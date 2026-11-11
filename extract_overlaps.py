import argparse
import logging
import pandas as pd
import os
import librosa 
import json
from tqdm import tqdm
from typing import List, Dict

from realtime_speech_separation.utils.data_utils import RAW_AUDIO_EXTENSIONS
from realtime_speech_separation.utils.audio_utils import save_audio

logger = logging.getLogger(__name__)

def load_vad(vad_path: str) -> List[Dict[str, float]]:
    try:
        with open(vad_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load VAD file {vad_path}: {e}")

def find_audio_ext(file_path: str) -> str:
    for ext in RAW_AUDIO_EXTENSIONS:
        file_with_ext = f"{file_path}{ext}"
        if os.path.exists(file_with_ext):
            return file_with_ext
    raise FileNotFoundError(f"No audio file found for {file_path} with supported extensions.")

def get_overlap_segments(
    vad_ch0: List[Dict[str, float]], 
    vad_ch1: List[Dict[str, float]], 
    overlap_vad_merge_secs: float = 1.0,
    min_overlap_secs: float = 1.0,
    pad_secs: float = 1.0,
) -> List[Dict[str, float]]:
    overlaps = []
    i, j = 0, 0
    while i < len(vad_ch0) and j < len(vad_ch1):
        start_ch0, end_ch0 = vad_ch0[i]["start"], vad_ch0[i]["end"]
        start_ch1, end_ch1 = vad_ch1[j]["start"], vad_ch1[j]["end"]
        start_overlap = max(start_ch0, start_ch1)
        end_overlap = min(end_ch0, end_ch1)
        if start_overlap < end_overlap:
            overlaps.append({"start": start_overlap, "end": end_overlap})
        if end_ch0 < end_ch1:
            i += 1
        else:
            j += 1

    # Merge consecutive overlap segments that are closer than `overlap_vad_merge_secs`
    merged_overlaps = []
    for overlap in overlaps:
        if merged_overlaps and overlap["start"] - merged_overlaps[-1]["end"] < overlap_vad_merge_secs:
            merged_overlaps[-1]["end"] = max(merged_overlaps[-1]["end"], overlap["end"])
        else:
            merged_overlaps.append(overlap)

    # Filter out overlap segments that are shorter than `min_overlap_secs` and apply padding
    filtered_overlaps = []
    for overlap in merged_overlaps:
        start, end = overlap["start"], overlap["end"]
        if end - start >= min_overlap_secs:
            filtered_overlaps.append({"start": max(0, start - pad_secs), "end": end + pad_secs})
    return filtered_overlaps

def main(args: argparse.Namespace):
    # Build split dict
    splits_df = pd.read_csv(args.file_splits_csv)

    # Get audio files
    audio_files = [
        os.path.join(args.raw_audio_dir, file_id)
        for file_id in splits_df[splits_df["split"] == args.split]["file_id"]
    ]

    # Process the audio files
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        audio_file = find_audio_ext(audio_file)
        base_file, _ = os.path.splitext(audio_file)
        output_base_file = base_file.replace(args.raw_audio_dir, args.output_dir)
        
        # Read the audio file
        audio, sr = librosa.load(audio_file, sr=None, mono=False)
        if audio.ndim == 1:
            logger.warning(f"Audio file {audio_file} is mono. Skipping.")
            continue
        # Read the corresponding VAD files for each channel
        vad_file_ch0 = base_file.replace(args.raw_audio_dir, args.vads_dir) + "_c0.json"
        vad_file_ch1 = base_file.replace(args.raw_audio_dir, args.vads_dir) + "_c1.json"
        vad_ch0 = load_vad(vad_file_ch0)
        vad_ch1 = load_vad(vad_file_ch1)
        # Compute the overlapping segments between the two VAD channels
        overlaps = get_overlap_segments(vad_ch0, vad_ch1, args.overlap_vad_merge_secs, args.min_overlap_secs, args.pad_secs)
        if not overlaps:
            logger.warning(f"No overlapping segments longer than {args.min_overlap_secs}s found for audio file {audio_file}.")
            continue
        # Extract and save each overlapping segment as a separate audio file
        for overlap in overlaps:
            start, end = overlap["start"], overlap["end"]
            overlap_audio = audio[..., int(start * sr):int(end * sr)]
            output_file = os.path.join(output_base_file, f"overlap_{start:.2f}_{end:.2f}.wav")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            save_audio(output_file, overlap_audio, sr)
            logger.info(f"Saved overlapping segment to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract samples of overlapping speech for codec evaluation.")

    input_group = parser.add_argument_group("Input Arguments")
    input_group.add_argument("--file_splits_csv", type=str, default="output/dataset_magicodec_131k_60s_file_splits.csv", help="CSV file containing the manifest of audio files and their split membership.")
    input_group.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"], help="Dataset split to use.")
    input_group.add_argument("--raw_audio_dir", type=str, default="data/audio/raw", help="Directory containing raw audio files referenced in the metadata.")
    input_group.add_argument("--vads_dir", type=str, default="data/vads", help="Directory containing VAD predictions for the audio files.")
    input_group.add_argument("--overlap_vad_merge_secs", type=float, default=1.0, help="Minimum gap between overlap segments to merge them.")
    input_group.add_argument("--min_overlap_secs", type=float, default=1.0, help="Minimum duration of overlap segments to keep.")
    input_group.add_argument("--pad_secs", type=float, default=1.0, help="Padding to apply to overlap segments.")

    output_group = parser.add_argument_group("Output Arguments")
    output_group.add_argument("--output_dir", type=str, default="output/codec_overlap_eval", help="Directory to save evaluation output audio.")

    logging_group = parser.add_argument_group("Logging Arguments")
    logging_group.add_argument("--log_level", type=str, default="WARNING", choices=list(logging.getLevelNamesMapping()), help="Logging level.")
    
    args = parser.parse_args()

    log_level = logging.getLevelNamesMapping().get(args.log_level.upper())
    logging.basicConfig(level=log_level)
    logger.info(f"Running with args: {args}")
    
    main(args)