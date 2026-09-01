import argparse
import os
import functools
import jsonlines
from tqdm import tqdm

from realtime_speech_separation.lm_dataset_builder import LMDatasetBuilder
from codec_bpe import UNICODE_OFFSET_LARGE
from codec_bpe.core.utils import get_codec_info, update_args_from_codec_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the speech separation training dataset"
    )
    parser.add_argument("--codes_path", type=str, required=True)
    parser.add_argument("--vads_path", type=str, default="data/vads")
    parser.add_argument("--num_codebooks", type=int, default=None)
    parser.add_argument("--codebook_size", type=int, default=None)
    parser.add_argument("--codec_framerate", type=float, default=None)
    parser.add_argument("--header_target_voice_token", type=str, default="<|target_voice|>")
    parser.add_argument("--header_end_token", type=str, default="<|end_header|>")
    # handle hex values for unicode_offset with argparse: https://stackoverflow.com/a/25513044
    parser.add_argument("--unicode_offset", type=functools.partial(int, base=0), default=UNICODE_OFFSET_LARGE)
    parser.add_argument("--context_secs", type=float, default=40.0)
    parser.add_argument("--overlap_secs", type=float, default=10.0)
    parser.add_argument("--max_voice_enrollment_secs", type=float, default=10.0)
    parser.add_argument("--voice_enrollment_vad_merge_secs", type=float, default=1.0)
    parser.add_argument("--voice_enrollment_selection_seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="output/lm_dataset.txt")
    parser.add_argument("--codes_filter", type=str, nargs="+")
    parser.add_argument("--codes_filter_exclude", type=str, nargs="+")
    parser.add_argument("--num_examples", type=int, default=None)
    args = parser.parse_args()

    codec_info = get_codec_info(args.codes_path)
    if not codec_info:
        # maybe the codec_info.json file is nested in the mono/stereo subdirectories
        codec_info = get_codec_info(os.path.join(args.codes_path, "stereo"))
    update_args_from_codec_info(args, codec_info)
    if args.num_codebooks is None or args.codebook_size is None or args.codec_framerate is None:
        raise ValueError(
            "codec_info.json does not exist in --codes_path so you must specify --num_codebooks, --codebook_size, and --codec_framerate manually."
        )

    lm_dataset_builder = LMDatasetBuilder(
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        codec_framerate=args.codec_framerate,
        header_target_voice_token=args.header_target_voice_token,
        header_end_token=args.header_end_token,
        unicode_offset=args.unicode_offset,
        context_secs=args.context_secs,
        overlap_secs=args.overlap_secs,
        max_voice_enrollment_secs=args.max_voice_enrollment_secs,
        voice_enrollment_vad_merge_secs=args.voice_enrollment_vad_merge_secs,
        voice_enrollment_selection_seed=args.voice_enrollment_selection_seed,
    )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    metadata_path = args.save_path.replace(".txt", "_metadata.jsonl")

    with open(args.save_path, "w", encoding="utf-8") as f:
        with jsonlines.open(metadata_path, "w") as f_meta:
            example_iterator = lm_dataset_builder.iterate_examples(
                args.codes_path, args.vads_path, args.codes_filter, args.codes_filter_exclude
            )
            for i, (example, metadata) in tqdm(enumerate(example_iterator), desc="Examples"):
                if i == args.num_examples:
                    break
                f.write(example)
                f.write("\n")

                f_meta.write(metadata)
            