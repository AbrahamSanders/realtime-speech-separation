import os
import shutil
import argparse
import librosa
import json
from datetime import datetime
from tqdm import tqdm
from silero_vad import load_silero_vad, get_speech_timestamps

SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".flac", ".opus"]
SAMPLE_RATE = 16000

class VADResult:
    def __init__(self):
        self.num_audio_files = 0
        self.num_vad_files = 0
        self.num_skipped_dirs = 0
        self.errored_audio_files = []

def main(args):
    model = load_silero_vad()
    result = VADResult()
    debug_break = False

    # iterate and convert
    start_time = datetime.now()
    for root, _, files in os.walk(args.audio_path):
        files = sorted([os.path.join(root, f) for f in files if os.path.splitext(f)[1] in args.extensions])
        if args.audio_filter:
            files = [f for f in files if any([filter_ in f for filter_ in args.audio_filter])]
        if len(files) == 0:
            continue
        vad_root = root.replace(args.audio_path, args.vad_path)
        if os.path.exists(vad_root):
            if args.overwrite:
                shutil.rmtree(vad_root)
            else:
                print(f"Skipping {root} because {vad_root} already exists.")
                result.num_skipped_dirs += 1
                continue
        print(f"Predicting VAD in {root}...")
        for file_path in tqdm(files, desc="Files"):
            result.num_audio_files += 1
            try:
                # Load the audio file
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=False)
                if audio.ndim == 1:
                    audio = audio.reshape(1, -1)  # Convert mono to 2D array with one channel
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                result.errored_audio_files.append(file_path)
                continue

            for channel in range(audio.shape[0]):
                speech_timestamps = get_speech_timestamps(audio[channel], model, sampling_rate=SAMPLE_RATE, return_seconds=True, time_resolution=2)
                file_name_noext = os.path.basename(os.path.splitext(file_path)[0])
                vad_filepath = os.path.join(vad_root, f"{file_name_noext}_c{channel}.json")
                os.makedirs(os.path.dirname(vad_filepath), exist_ok=True)
                with open(vad_filepath, "w") as f:
                    json.dump(speech_timestamps, f, indent=4)
                result.num_vad_files += 1

            if args.debug_num_files and result.num_audio_files >= args.debug_num_files:
                debug_break = True
                break
        if debug_break:
            break

    end_time = datetime.now()

    # Print summary
    print(f"Attempted to predict VAD in {result.num_audio_files} audio files:")
    print(f"{result.num_audio_files-len(result.errored_audio_files)} Succeeded.")
    print(f"{len(result.errored_audio_files)} Errored.")
    print(f"{result.num_vad_files} VAD files created.")
    print(f"{result.num_skipped_dirs} directories skipped.")
    if result.errored_audio_files:
        print("\nErrored files:")
        for file in result.errored_audio_files:
            print(file)
    print(f"Processing completed in {end_time - start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare VAD results needed to select channel-specific target voice samples for use during training"
    )
    parser.add_argument("--audio_path", type=str, default="data/audio/raw", help="Directory containing the audio files")
    parser.add_argument("--vad_path", type=str, default="data/vad", help="Directory to save the VAD results")
    parser.add_argument("--extensions", nargs="+", default=SUPPORTED_EXTENSIONS,
        help="Audio file extensions to convert. Formats must be supported by a librosa backend.",
    )
    parser.add_argument("--audio_filter", nargs="+", 
        help="Audio file filters. If provided, file paths must match one of the filters to be converted.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing VAD files")
    parser.add_argument("--debug_num_files", type=int, help="Limit the number of audio files to process for debugging purposes")
    args = parser.parse_args()

    main(args)
