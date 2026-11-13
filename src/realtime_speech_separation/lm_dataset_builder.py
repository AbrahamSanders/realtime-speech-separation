from typing import Optional, Union, Iterator, List, Tuple, Dict
from tqdm import tqdm
import numpy as np
import itertools
import re
import random
import os
import json

from codec_bpe import codes_to_chars, UNICODE_OFFSET_LARGE
from codec_bpe.core.utils import get_codes_files

DELAY_TOKENS: List[str] = [
    "<|delay_0ms|>",
    "<|delay_100ms|>",
    "<|delay_200ms|>",
    "<|delay_400ms|>",
    "<|delay_800ms|>",
    "<|delay_1600ms|>",
    "<|delay_3200ms|>",
    "<|delay_6400ms|>",
]

class LMDatasetBuilder:
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        codec_framerate: float,
        header_delay_tokens: List[str] = DELAY_TOKENS,
        header_target_voice_token: str = "<|target_voice|>",
        header_end_token: str = "<|end_header|>",
        skip_char: str = "-",
        unicode_offset: int = UNICODE_OFFSET_LARGE,
        context_secs: float = 60.0,
        overlap_secs: float = 45.0,
        voice_enrollment_max_secs: float = 10.0,
        voice_enrollment_ideal_min_secs: float = 3.0,
        voice_enrollment_hard_min_secs: float = 2.0,
        voice_enrollment_vad_merge_secs: float = 1.0,
        voice_enrollment_ideal_min_candidates: int = 10,
        voice_enrollment_selection_seed: int = 42 ** 2,
        delay_selection_seed: int = 42 ** 3,
    ):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codec_framerate = codec_framerate
        self.header_delay_tokens = header_delay_tokens
        self.header_target_voice_token = header_target_voice_token
        self.header_end_token = header_end_token
        self.skip_char = skip_char
        self.unicode_offset = unicode_offset
        self.context_secs = context_secs
        self.overlap_secs = overlap_secs
        self.voice_enrollment_max_secs = voice_enrollment_max_secs
        self.voice_enrollment_ideal_min_secs = voice_enrollment_ideal_min_secs
        self.voice_enrollment_hard_min_secs = voice_enrollment_hard_min_secs
        self.voice_enrollment_vad_merge_secs = voice_enrollment_vad_merge_secs
        self.voice_enrollment_ideal_min_candidates = voice_enrollment_ideal_min_candidates
        self.voice_enrollment_selection_seed = voice_enrollment_selection_seed
        self.delay_selection_seed = delay_selection_seed

        self.context_codes = int(self.context_secs * self.codec_framerate * self.num_codebooks)
        self.overlap_codes = int(self.overlap_secs * self.codec_framerate * self.num_codebooks)
        if self.context_codes % self.num_codebooks != 0 or self.overlap_codes % self.num_codebooks != 0:
            raise ValueError(
                f"context_codes and overlap_codes must be divisible by {self.num_codebooks} "
                "To ensure examples do not start or end in the middle of an acoustic unit."
            )
        self.delay_info = {}
        for delay_token in self.header_delay_tokens:
            delay_secs = int(re.search(r"delay_(\d+)ms", delay_token).group(1)) / 1000.0
            if delay_secs >= self.context_secs:
                raise ValueError(
                    f"Delay {delay_token} ({delay_secs}s) must be smaller than the context duration ({self.context_secs}s)."
                )
            delay_codes = int(delay_secs * self.codec_framerate * self.num_codebooks)
            self.delay_info[delay_token] = (delay_secs, delay_codes)

        self.voice_enroll_rng_gen = random.Random(self.voice_enrollment_selection_seed)
        self.delay_rng_gen = random.Random(self.delay_selection_seed)

    def _load_vad(self, vad_path: str) -> List[Dict[str, float]]:
        try:
            with open(vad_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load VAD file {vad_path}: {e}")

    def _group_codes_files(self, codes_path: str, codes_files: List[str]) -> List[Tuple[str, List[List[str]]]]:
        grouped_codes_files = []
        last_file_root = None
        for codes_file in codes_files:
            codes_file_info = re.match(r"(.+)_c(\d+)[_.]", codes_file)
            if not codes_file_info:
                raise ValueError(
                    f"Invalid codes file name format: {codes_file}. Expected format: *_c<channel>.npy or *_c<channel>_<timestamp>.npy"
                )
            file_root, channel = codes_file_info.group(1), int(codes_file_info.group(2))
            # make file_root relative to the codes_path
            file_root = os.path.relpath(file_root, codes_path)
            if file_root != last_file_root:
                grouped_codes_files.append((file_root, []))
                last_file_root = file_root
            grouped_codes_files[-1][1].append((codes_file, channel))

        # separate the files in each groups by channel
        channel_grouped_codes_files = []
        for file_root, file_group in grouped_codes_files:
            num_channels = max([channel for _, channel in file_group]) + 1
            channel_grouped_codes_files.append(
                (
                    file_root, 
                    [[f[0] for f in file_group if f[1] == c] for c in range(num_channels)],
                )
            )

        return channel_grouped_codes_files

    def _filter_and_group_codes_files(
        self, 
        codes_path: str,
        codes_filter: Optional[Union[str, List[str]]] = None,
        codes_filter_exclude: Optional[Union[str, List[str]]] = None,
    ) -> List[Tuple[str, List[List[str]]]]:
        # get the codes files
        codes_files = get_codes_files(codes_path, codes_filter)
        if codes_filter_exclude:
            if isinstance(codes_filter_exclude, str):
                codes_filter_exclude = [codes_filter_exclude]
            codes_files = [f for f in codes_files if not any(ex in f for ex in codes_filter_exclude)]
        # group codes files by root filename (minus channel and starting timestamp) and then by channel
        grouped_codes_files = self._group_codes_files(codes_path, codes_files)
        return grouped_codes_files

    def _merge_grouped_codes_files(
        self, 
        grouped_codes_files_mono: List[Tuple[str, List[List[str]]]],
        grouped_codes_files_stereo: List[Tuple[str, List[List[str]]]],
    ) -> List[Tuple[str, List[List[str]]]]:
        grouped_codes_files = []
        mono_dict = dict(grouped_codes_files_mono)
        for file_root, stereo_channels in grouped_codes_files_stereo:
            if file_root in mono_dict:
                grouped_codes_files.append(
                    (file_root, mono_dict[file_root] + stereo_channels)
                )
            else:
                print(f"{file_root} is in stereo but not in mono. Skipping file...")
        return grouped_codes_files

    def _build_example_codes_str(
        self, 
        channels_chars: List[str], 
        example_start_code: int,
        example_end_code: int,
        target_channel: int,
        delay_codes: int,
    ) -> str:
        # sanity check: example must be longer than the delay
        if example_end_code-example_start_code <= delay_codes:
            raise ValueError(
                "Example length must be greater than the delay. This should have never happened - "
                "something went wrong in the main loop that slices the example ranges."
            )
        # sanity check: we have at least `delay_codes` available future codes
        if example_end_code+delay_codes > len(channels_chars[0]):
            raise ValueError(
                "Not enough future codes available for the specified delay. This should have never happened - "
                "something went wrong in the main loop that slices the example ranges."
            )

        # slice codes from the full sequence for the current example
        example_channels_chars = [ch[example_start_code:example_end_code] for ch in channels_chars]
        # if we are under a delay condition, shift the channels so that each pair of stereo codes correspond to 
        # the mono code from `delay_secs` seconds ago
        if delay_codes > 0:
            # Append `delay_codes` worth of future codes beyond `example_end_code` to the end of the mono channel, and
            # pad the beginning of the stereo channels with `delay_codes` worth of skip chars.
            # The end result has the mono inputs starting at `example_start_code` and the stereo outputs ending at `example_end_code`,
            # while maintaining the delayed alignment between mono and stereo channels.
            future_codes = channels_chars[0][example_end_code:example_end_code+delay_codes]
            example_channels_chars[0] += future_codes
            for i in range(1, len(example_channels_chars)):
                example_channels_chars[i] = self.skip_char * delay_codes + example_channels_chars[i]

        # reverse the stereo channels if we're targeting stereo channel 1
        if target_channel == 1:
            example_channels_chars = example_channels_chars[:1] + example_channels_chars[1:][::-1]

        # compile the codes string by interleaving the channels as M L R M L R ...
        codes_str = "".join(list(itertools.chain.from_iterable(zip(*example_channels_chars))))
        return codes_str

    def _preprocess_vad_for_target_voice_selection(
        self, 
        vad_data: List[Dict[str, float]], 
        target_channel_chars: str,
    ) -> List[Tuple[int, int, float, str]]:
        # Merge consecutive VAD segments that are closer than `voice_enrollment_vad_merge_secs`
        merged_vad_data = []
        for segment in vad_data:
            if merged_vad_data and segment["start"] - merged_vad_data[-1]["end"] < self.voice_enrollment_vad_merge_secs:
                merged_vad_data[-1]["end"] = max(merged_vad_data[-1]["end"], segment["end"])
            else:
                merged_vad_data.append(segment)

        # split VAD segments that are longer than `voice_enrollment_max_secs` into smaller segments
        final_vad_data = []
        for segment in merged_vad_data:
            start = segment["start"]
            end = segment["end"]
            while end - start > self.voice_enrollment_max_secs:
                final_vad_data.append({"start": start, "end": start + self.voice_enrollment_max_secs})
                start += self.voice_enrollment_max_secs
            if end - start > 0:
                final_vad_data.append({"start": start, "end": end})

        # format as speech ranges tuple with code indices, duration and speech chars
        speech_ranges = [
            (
                int(r["start"] * self.codec_framerate * self.num_codebooks),
                int(r["end"] * self.codec_framerate * self.num_codebooks),
                r["end"] - r["start"],
            )
            for r in final_vad_data
        ]
        # filter out speech ranges that are shorter than `voice_enrollment_hard_min_secs`
        # and extract the corresponding speech characters from the target channel
        speech_ranges = [
            (start_code, end_code, length_secs, target_channel_chars[start_code:end_code]) 
            for start_code, end_code, length_secs in speech_ranges
            if length_secs >= self.voice_enrollment_hard_min_secs
        ]

        # sort by duration descending
        speech_ranges.sort(key=lambda x: x[2], reverse=True)

        return speech_ranges

    def _select_target_voice(
        self,
        target_channel_speech_ranges: List[Tuple[int, int, float, str]],
        example_start_code: int,
        example_end_code: int,
    ) -> Optional[Tuple[int, int, str]]:
        # only sample from speech segments outside the current example range.
        target_channel_speech_ranges = [
            (start_code, end_code, length_secs, voice_str) for start_code, end_code, length_secs, voice_str in target_channel_speech_ranges 
            if (end_code <= example_start_code or start_code >= example_end_code) # outside the current example range
        ]
        # take `voice_enrollment_ideal_min_candidates` longest candidates or all that are
        # `voice_enrollment_ideal_min_secs` and longer, whichever yields more candidates.
        voice_candidates = [
            (start_code, end_code, voice_str) for i, (start_code, end_code, length_secs, voice_str) in enumerate(target_channel_speech_ranges) 
            if i < self.voice_enrollment_ideal_min_candidates or length_secs >= self.voice_enrollment_ideal_min_secs
        ]
        # Select a random voice candidate from voice_candidates
        if not voice_candidates:
            return None
        selected_voice = self.voice_enroll_rng_gen.choice(voice_candidates)
        return selected_voice

    def iterate_examples(
        self, 
        codes_path: str,
        vads_path: str,
        codes_filter: Optional[Union[str, List[str]]] = None,
        codes_filter_exclude: Optional[Union[str, List[str]]] = None,
    ) -> Iterator[str]:
        self.delay_rng_gen.seed(self.delay_selection_seed)

        # get the mono and stereo codes files
        grouped_codes_files_mono = self._filter_and_group_codes_files(os.path.join(codes_path, "mono"), codes_filter, codes_filter_exclude)
        grouped_codes_files_stereo = self._filter_and_group_codes_files(os.path.join(codes_path, "stereo"), codes_filter, codes_filter_exclude)
        # merge channels: mono first, then the two stereo channels. Drop any file roots that are not in both mono and stereo lists
        grouped_codes_files = self._merge_grouped_codes_files(grouped_codes_files_mono, grouped_codes_files_stereo)

        # sanity check: skip all grouped codes files that don't have the expected number of channels
        grouped_codes_files_verified = []
        for file_root, file_channels in grouped_codes_files:
            if len(file_channels) != 3:
                print(f"Expected 3 channels (1 mono + 2 stereo) for {file_root}, but got {len(file_channels)}. Skipping file...")
            else:
                grouped_codes_files_verified.append((file_root, file_channels))
        
        # iterate over each group of codes files
        for file_root, file_channels in tqdm(grouped_codes_files_verified, desc="Codes file groups"):
            # reseed the voice enrollment random number generator for each file root, this way we get the same 
            # voice enrollment selections per file even if the number/order of files in the dataset changes.
            self.voice_enroll_rng_gen.seed(self.voice_enrollment_selection_seed)

            # concatenate all codes files in each group for each channel
            codes = np.stack(
                [
                    np.concatenate([np.load(file) for file in file_group], axis=-1) 
                    for file_group in file_channels
                ], 
                axis=0,
            )
            if len(codes.shape) == 5:
                codes = codes[:, 0, 0]
            elif len(codes.shape) == 4:
                codes = codes[:, 0]
            codes = codes[:, :self.num_codebooks] # shape: (num_channels, num_codebooks, sequence_length)

            # convert codes to unicode string
            channels_chars = [
                codes_to_chars(
                    ch_codes, 
                    self.codebook_size, 
                    copy_before_conversion=False,
                    unicode_offset=self.unicode_offset,
                ) for ch_codes in codes
            ]

            # If this is a "one-vs-all" sample where channel 0 has the target speaker and channel 1 has multiple mixed other speakers,
            # we'll only generate examples with channel 0 as the target speaker.
            # If this is a "one-vs-one" sample where each channel has a single speaker,
            # we'll generate two examples per delay setting, one for each channel as the target speaker.
            target_channels = [0] if "one_vs_all" in file_root.lower() else [0, 1]

            # build the examples
            for target_channel in target_channels:
                # load the speech ranges for the target channel for use in selecting target voices for each example
                target_channel_vad_file = os.path.join(vads_path, f"{file_root}_c{target_channel}.json")
                target_channel_vad = self._load_vad(target_channel_vad_file)
                target_channel_chars = channels_chars[1 + target_channel]
                target_channel_speech_ranges = self._preprocess_vad_for_target_voice_selection(target_channel_vad, target_channel_chars)
                if not target_channel_speech_ranges:
                    print(
                        f"No speech ranges longer than {self.voice_enrollment_hard_min_secs} seconds found for "
                        f"target channel {target_channel} in file {file_root}. Skipping channel..."
                    )
                    continue

                # yield examples from the sequence with the specified context length and overlap
                start_code = 0
                example_index = 0
                while True:
                    delay_token = self.delay_rng_gen.choice(self.header_delay_tokens)
                    delay_secs, delay_codes = self.delay_info[delay_token]

                    end_code = min(start_code + self.context_codes, len(channels_chars[0])-delay_codes)
                    if end_code-start_code <= delay_codes:
                        print(
                            f"Example in range {start_code}-{end_code} for target channel {target_channel} "
                            f"in file {file_root} is too short for delay {delay_secs}. Skipping last example..."
                        )
                        break

                    # select the target voice for the current example
                    target_voice = self._select_target_voice(target_channel_speech_ranges, start_code, end_code+delay_codes)
                    if target_voice is None:
                        print(
                            f"Could not find target voice sample outside of range {start_code}-{end_code+delay_codes} "
                            f"on channel {target_channel} for delay {delay_secs} in file {file_root}. Skipping example..."
                        )
                    else:
                        # build & yield the example and metadata
                        tv_start_code, tv_end_code, tv_str = target_voice
                        metadata = {
                            "file_id": file_root,
                            "target_channel": target_channel,
                            "example_index": example_index,
                            "ex_start_secs": start_code / (self.codec_framerate * self.num_codebooks),
                            "ex_end_secs": end_code / (self.codec_framerate * self.num_codebooks),
                            "delay_secs": delay_secs,
                            "tv_start_secs": tv_start_code / (self.codec_framerate * self.num_codebooks),
                            "tv_end_secs": tv_end_code / (self.codec_framerate * self.num_codebooks),
                        }
                        example = self._build_example_codes_str(channels_chars, start_code, end_code, target_channel, delay_codes)
                        example = f"{delay_token}{self.header_target_voice_token}{tv_str}{self.header_end_token}{example}"
                        yield example, metadata
                        example_index += 1

                    # move to next start_code, or break if we've reached the end of the sequence
                    if end_code >= len(channels_chars[0])-delay_codes:
                        break
                    start_code = end_code - self.overlap_codes