import argparse
import logging
import jsonlines
import os
import random
from tqdm import tqdm

from realtime_speech_separation.utils.data_utils import try_load_original_audio
from realtime_speech_separation.utils.audio_utils import save_audio

logger = logging.getLogger(__name__)

def main(args: argparse.Namespace):
    metadata_name = os.path.splitext(os.path.basename(args.metadata))[0]
    output_dir = os.path.join(args.output_dir, metadata_name)
   
    # get metadata list
    with jsonlines.open(args.metadata) as reader:
        metadata_list = list(reader)

    # sample the list if sampling_p is less than 1.0
    if args.sampling_p < 1.0:
        logger.info(f"Sampling {args.sampling_p * 100}% of the metadata list with seed {args.sampling_seed}.")
        random.seed(args.sampling_seed)
        metadata_list = random.sample(metadata_list, int(len(metadata_list) * args.sampling_p))
    
    # process all metadata entries
    for metadata in tqdm(metadata_list, desc="Processing metadata"):
        file_base = os.path.join(output_dir, metadata["file_id"], f"target_channel_{metadata['target_channel']}", str(metadata["example_index"]))
        input_file = f"{file_base}.wav"
        enrollment_file = f"{file_base}_enrollment.wav"
        reference_file = f"{file_base}_reference.wav"

        if os.path.exists(input_file) and os.path.exists(enrollment_file) and os.path.exists(reference_file):
            logger.info(f"All inference input files for file_id {metadata['file_id']} already exist. Skipping.")
            continue

        # Load original audio (enrollment, input, reference) for the given metadata
        orig_audio = try_load_original_audio(metadata, args.raw_audio_dir)
        if orig_audio is None:
            logger.warning(f"Original audio not found for file_id {metadata['file_id']}. Skipping.")
            return None
        enrollment_audio, input_audio, reference_audio, sr = orig_audio

        # Save input, enrollment, and reference files to the output directory for easy access by eval scripts
        # without needing to reload the metadata
        os.makedirs(os.path.dirname(file_base), exist_ok=True)
        
        save_audio(input_file, input_audio, sr)
        logger.info(f"Input audio written to: {input_file}")
        
        save_audio(enrollment_file, enrollment_audio, sr)
        logger.info(f"Enrollment audio written to: {enrollment_file}")

        save_audio(reference_file, reference_audio, sr)
        logger.info(f"Reference audio written to: {reference_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract inference inputs for evaluation from the dataset metadata.")

    input_group = parser.add_argument_group("Input Arguments")
    input_group.add_argument("--metadata", type=str, required=True, help="Metadata .jsonl file containing the examples for evaluation.")
    input_group.add_argument("--raw_audio_dir", type=str, default="data/audio/raw", help="Directory containing raw audio files referenced in the metadata.")

    output_group = parser.add_argument_group("Output Arguments")
    output_group.add_argument("--output_dir", type=str, default="output/eval", help="Directory to save evaluation output audio.")
    output_group.add_argument("--sampling_p", type=float, default=1.0, help="Proportion of the dataset to extract for evaluation (0.0, 1.0].")
    output_group.add_argument("--sampling_seed", type=int, default=42, help="Random seed for sampling the dataset.")

    logging_group = parser.add_argument_group("Logging Arguments")
    logging_group.add_argument("--log_level", type=str, default="WARNING", choices=list(logging.getLevelNamesMapping()), help="Logging level.")
    
    args = parser.parse_args()

    if args.sampling_p <= 0.0 or args.sampling_p > 1.0:
        parser.error("sampling_p must be between 0.0 and 1.0, not including 0.0.")

    log_level = logging.getLevelNamesMapping().get(args.log_level.upper())
    logging.basicConfig(level=log_level)
    logger.info(f"Running with args: {args}")
    
    main(args)