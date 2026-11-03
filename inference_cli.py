from typing import Tuple, List
import numpy as np
import argparse
import logging
import librosa
import os
import soundfile as sf
from tqdm import tqdm, trange

from realtime_speech_separation import RealtimeSeparator, RealtimeSeparatorResources, RealtimeSeparatorConfig, add_common_inference_args
from realtime_speech_separation.utils.audio_utils import pad_or_trim, preprocess_enrollment_audio
from realtime_speech_separation.realtime_separator_config import SUPPORTED_DELAY_SECS

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".flac", ".opus"]

def get_audio_files(args: argparse.Namespace) -> List[Tuple[str, str]]:
    audio_files = []
    if args.input_file:
        audio_files.append((args.input_file, args.enrollment_file))
    elif args.input_dir:
        for file_name in os.listdir(args.input_dir):
            file_name, ext = os.path.splitext(file_name)
            if ext.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if file_name.lower().endswith("_enrollment"):
                continue
            if file_name.lower().endswith("_separated"):
                continue
            input_file = os.path.join(args.input_dir, f"{file_name}{ext}")
            enrollment_file = os.path.join(args.input_dir, f"{file_name}_enrollment{ext}")
            audio_files.append((input_file, enrollment_file))
    return audio_files

def main(args: argparse.Namespace):
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

    audio_files = get_audio_files(args)
    sr = separator.resources.mono_audio_tokenizer.sampling_rate
    for input_file, enrollment_file in tqdm(audio_files, desc="Processing audio files"):
        logger.info(f"Processing input audio: {input_file}")
        # Check if enrollment audio exists and load/preprocess it if available
        if enrollment_file is None or not os.path.exists(enrollment_file):
            separator.config.target_voice_enrollment = None
            logger.warning(
                f"Enrollment audio not provided for input audio {input_file}. A 10 second silent enrollment will be used. "
                "This may cause unexpected separation results."
            )
        else:
            enrollment_audio, _ = librosa.load(enrollment_file, sr=sr, mono=True, dtype=np.float32)
            separator.config.target_voice_enrollment = preprocess_enrollment_audio((sr, enrollment_audio), sr, return_tuple=True)

        # Reset separator context & apply the new enrollment
        separator.reset()

        # Load the input audio & prepare it for separation
        input_audio, _ = librosa.load(input_file, sr=sr, mono=True, dtype=np.float32)
        input_audio_length = input_audio.shape[-1]
        # If there is a delay, pad the input audio with silence at the end to make sure we stream the entire output
        if separator.config.delay_secs > 0.0:
            delay_samples = int(separator.config.delay_secs * sr)
            input_audio = np.concatenate([input_audio, np.zeros(delay_samples, dtype=np.float32)])

        # Do the separation
        for start in trange(0, input_audio.shape[-1], separator.chunk_size_samples, desc=f"Processing {os.path.basename(input_file)}"):
            end = start + separator.chunk_size_samples
            chunk = input_audio[start:end]
            chunk = pad_or_trim(chunk, separator.chunk_size_samples)
            _ = separator.process_audio(chunk)

        # Output the results
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
        output_file = os.path.splitext(input_file)[0] + "_separated.wav"
        sf.write(output_file, output_audio.T, sr)
        logger.info(f"Separated audio written to: {output_file}")
        separator.profilers.log(tag=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the Realtime Speech Separation CLI.")
    add_common_inference_args(parser)
    default_config = RealtimeSeparatorConfig()
    parser.add_argument("--input_file", type=str, default=None, help="Path to the enrollment audio file.")
    parser.add_argument("--input_dir", type=str, default=None, help="Path to the input directory containing input/enrollment pairs.")
    parser.add_argument("--enrollment_file", type=str, default=None, help="Path to the enrollment audio file.")
    parser.add_argument("--delay_secs", type=float, default=default_config.delay_secs, help="Delay (seconds).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for random)")
    parser.add_argument("--chunk_size_secs", type=float, default=1.0, help="Chunk size (seconds).")
    parser.add_argument("--temperature", type=float, default=default_config.temperature, help="Temperature.")
    parser.add_argument("--top_k", type=int, default=default_config.top_k, help="Top-k.")
    parser.add_argument("--top_p", type=float, default=default_config.top_p, help="Top-p.")
    parser.add_argument("--min_p", type=float, default=default_config.min_p, help="Min-p.")
    parser.add_argument("--repeat_penalty", type=float, default=default_config.repeat_penalty, help="Repeat Penalty.")
    parser.add_argument("--presence_penalty", type=float, default=default_config.presence_penalty, help="Presence Penalty.")
    parser.add_argument("--frequency_penalty", type=float, default=default_config.frequency_penalty, help="Frequency Penalty.")
    parser.add_argument("--max_context_secs", type=float, default=default_config.max_context_secs, help="Max Context Length (seconds).")
    parser.add_argument("--trim_by_secs", type=float, default=default_config.trim_by_secs, help="Trim By (seconds).")
    parser.add_argument("--run_profilers", action="store_true", help="Run Profilers.")
    parser.add_argument("--log_level", type=str, default="WARNING", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    
    args = parser.parse_args()

    if not args.input_file and not args.input_dir:
        parser.error("You must specify either --input_file to separate a single audio file or --input_dir to separate a directory of audio files.")
    if args.input_file and args.input_dir:
        parser.error("You cannot specify both --input_file and --input_dir at the same time.")
    if args.input_dir and args.enrollment_file:
        parser.error("You cannot specify --enrollment_file when using --input_dir. The enrollments should be paired with the input files in the directory.")
    if not args.delay_secs in SUPPORTED_DELAY_SECS:
        parser.error(f"Unsupported delay_secs value: {args.delay_secs}. Supported values are: {SUPPORTED_DELAY_SECS}")
    log_level = logging.getLevelNamesMapping().get(args.log_level.upper())
    if log_level is None:
        parser.error(f"Invalid log level: {args.log_level}. Supported values are: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    logging.basicConfig(level=log_level)
    logger.info(f"Running with args: {args}")

    if args.run_profilers and log_level > logging.INFO:
        logger.warning("Profilers are enabled but the log level is higher than INFO. Profiler outputs will not be visible.")
    
    main(args)