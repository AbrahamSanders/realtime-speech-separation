from typing import Tuple, Optional
import numpy as np
import argparse
import logging
import librosa
import os
from tqdm import tqdm, trange

from realtime_speech_separation import RealtimeSeparator, RealtimeSeparatorResources, RealtimeSeparatorConfig, add_common_inference_args
from realtime_speech_separation.utils.audio_utils import pad_or_trim, preprocess_enrollment_audio, save_audio
from realtime_speech_separation.realtime_separator_config import SUPPORTED_DELAY_SECS
from realtime_speech_separation.utils.data_utils import get_audio_files
from realtime_speech_separation.utils.cli_utils import get_inference_args_hash

logger = logging.getLogger(__name__)

def get_enrollment_file(input_file: str) -> str:
    file_name, ext = os.path.splitext(input_file)
    return f"{file_name}_enrollment{ext}"

def get_output_file(input_file: str, output_format: str, args_hash: str) -> str:
    file_name, _ = os.path.splitext(input_file)
    return f"{file_name}_separated_{args_hash}.{output_format}"

def load_audio_file(input_file: str, sr: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    input_audio, _ = librosa.load(input_file, sr=sr, mono=True, dtype=np.float32)
    enrollment_file = get_enrollment_file(input_file)
    if not os.path.exists(enrollment_file):
        enrollment_audio = None
        logger.warning(
            f"Enrollment audio {enrollment_file} not found. A 10 second silent enrollment will be used. "
            "This may cause unexpected separation results."
        )
    else:
        enrollment_audio, _ = librosa.load(enrollment_file, sr=sr, mono=True, dtype=np.float32)
    return input_audio, enrollment_audio

def main(args: argparse.Namespace, args_hash: str):
    separator = RealtimeSeparator(
        resources=RealtimeSeparatorResources(
            llm_model_path=args.llm_model_path,
        ),
        config=RealtimeSeparatorConfig(
            delay_secs=args.delay_secs,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repeat_penalty=args.repeat_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            chunk_size_secs=args.chunk_size_secs,
            max_context_secs=args.max_context_secs,
            trim_by_secs=args.trim_by_secs,
            seed=args.seed,
            run_profilers=args.run_profilers
        ),
    )

    sr = separator.resources.mono_audio_tokenizer.sampling_rate
    input_files = get_audio_files(
        args.input_dir, 
        filter_include=args.input_filter, 
        filter_exclude=["_enrollment.", "_separated_", "_reference."],
    )

    for input_file in tqdm(input_files, desc="Processing input files"):
        output_file = get_output_file(input_file, args.output_format, args_hash)
        if os.path.exists(output_file) and not args.overwrite:
            logger.info(f"Output file {output_file} already exists. Skipping.")
            continue

        logger.info(f"Processing input audio: {input_file}")

        # Load the input audio and corresponding enrollment audio
        input_audio, enrollment_audio = load_audio_file(input_file, sr)
        
        # Apply the new enrollment & Reset separator context
        if enrollment_audio is not None:
            separator.config.target_voice_enrollment = preprocess_enrollment_audio((sr, enrollment_audio), sr, return_tuple=True)
        else:
            separator.config.target_voice_enrollment = None
        separator.reset()

        # Prepare input audio for separation
        input_audio_length = input_audio.shape[-1]
        # If there is a delay, pad the input audio with silence at the end to make sure we stream the entire output
        if separator.config.delay_secs > 0.0:
            delay_samples = int(separator.config.delay_secs * sr)
            input_audio = np.concatenate([input_audio, np.zeros(delay_samples, dtype=np.float32)])

        # Do the separation
        input_file_rel = os.path.relpath(input_file, args.input_dir)
        for start in trange(0, input_audio.shape[-1], separator.chunk_size_samples, desc=f"Processing {input_file_rel}"):
            end = start + separator.chunk_size_samples
            chunk = input_audio[start:end]
            chunk = pad_or_trim(chunk, separator.chunk_size_samples)
            _ = separator.process_audio(chunk)

        # Prepare to output the results
        output_audio = separator.get_stereo_audio_history()
        # If there is a delay, trim the silence at the front of the output audio
        if separator.config.delay_secs > 0.0:
            output_audio = output_audio[..., delay_samples:]
        # trim any silence from final chunk padding at the end of the output audio
        output_audio = output_audio[..., :input_audio_length]
        # Verify that the output audio is the same length as the input audio.
        # This will always be true unless the output is somehow shorter than the input, which should never happen.
        # Nonetheless, we want to check for this because it would indicate something is wrong with the separation process.
        if output_audio.shape[-1] != input_audio_length:
            logger.warning(
                f"Output audio length ({output_audio.shape[-1]} samples; {(output_audio.shape[-1] / sr):.3f} secs) "
                f"does not match input audio length ({input_audio_length} samples; {(input_audio_length / sr):.3f} secs)."
            )

        # Save the separated audio
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_audio(output_file, output_audio, sr)
        logger.info(f"Separated audio written to: {output_file}")

        # Log profiler metrics for this output file
        separator.profilers.log(tag=output_file)

if __name__ == "__main__":
    default_config = RealtimeSeparatorConfig()

    parser = argparse.ArgumentParser("Run the Realtime Speech Separation CLI.")
    add_common_inference_args(parser)

    input_group = parser.add_argument_group("Input Arguments")
    input_group.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing input/enrollment pairs.")
    input_group.add_argument("--input_filter", type=str, nargs="+", help="Filter for input files.")

    inference_group = parser.add_argument_group("Inference Arguments")
    inference_group.add_argument("--delay_secs", type=float, default=default_config.delay_secs, choices=SUPPORTED_DELAY_SECS, help="Delay (seconds).")
    inference_group.add_argument("--seed", type=int, default=None, help="Random seed (omit for random).")
    inference_group.add_argument("--chunk_size_secs", type=float, default=1.0, help="Chunk size (seconds).")
    inference_group.add_argument("--temperature", type=float, default=default_config.temperature, help="Temperature.")
    inference_group.add_argument("--top_k", type=int, default=default_config.top_k, help="Top-k.")
    inference_group.add_argument("--top_p", type=float, default=default_config.top_p, help="Top-p.")
    inference_group.add_argument("--min_p", type=float, default=default_config.min_p, help="Min-p.")
    inference_group.add_argument("--repeat_penalty", type=float, default=default_config.repeat_penalty, help="Repeat Penalty.")
    inference_group.add_argument("--presence_penalty", type=float, default=default_config.presence_penalty, help="Presence Penalty.")
    inference_group.add_argument("--frequency_penalty", type=float, default=default_config.frequency_penalty, help="Frequency Penalty.")
    inference_group.add_argument("--max_context_secs", type=float, default=default_config.max_context_secs, help="Max Context Length (seconds).")
    inference_group.add_argument("--trim_by_secs", type=float, default=default_config.trim_by_secs, help="Trim By (seconds).")

    output_group = parser.add_argument_group("Output Arguments")
    output_group.add_argument("--output_format", type=str, default="mp3", choices=["wav", "mp3"], help="Output audio format (wav, mp3).")
    output_group.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")

    logging_group = parser.add_argument_group("Logging Arguments")
    logging_group.add_argument("--run_profilers", action="store_true", help="Run Profilers.")
    logging_group.add_argument("--log_level", type=str, default="WARNING", choices=list(logging.getLevelNamesMapping()), help="Logging level.")
    
    args = parser.parse_args()

    log_level = logging.getLevelNamesMapping().get(args.log_level.upper())
    logging.basicConfig(level=log_level)
    logger.info(f"Running with args: {args}")

    if args.run_profilers and log_level > logging.INFO:
        logger.warning("Profilers are enabled but the log level is higher than INFO. Profiler outputs will not be visible.")

    if args.seed is None:
        logger.warning("No seed provided. Results will be non-deterministic.")

    args_hash = get_inference_args_hash(parser, args)

    main(args, args_hash)