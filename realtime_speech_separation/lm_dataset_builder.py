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

class LMDatasetBuilder:
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        codec_framerate: float,
        header_target_voice_token: str = "<|target_voice|>",
        header_end_token: str = "<|end_header|>",
        unicode_offset: int = UNICODE_OFFSET_LARGE,
        context_secs: float = 40.0,
        overlap_secs: float = 10.0,
        max_voice_enrollment_secs: float = 10.0,
        voice_enrollment_vad_merge_secs: float = 1.0,
        voice_enrollment_selection_seed: int = 42,
    ):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codec_framerate = codec_framerate
        self.unicode_offset = unicode_offset
        self.context_secs = context_secs
        self.overlap_secs = overlap_secs
        self.max_voice_enrollment_secs = max_voice_enrollment_secs
        self.voice_enrollment_vad_merge_secs = voice_enrollment_vad_merge_secs
        self.voice_enrollment_selection_seed = voice_enrollment_selection_seed

        self.header_target_voice_token = header_target_voice_token
        self.header_end_token = header_end_token

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

    def _build_codes_str(self, channels_chars: List[str]) -> str:
        # compile the codes string
        codes_str = "".join(list(itertools.chain.from_iterable(zip(*channels_chars))))
        return codes_str

    def _preprocess_vad_for_target_voice_selection(
        self, 
        vad_data: List[Dict[str, float]], 
        target_channel_chars: str,
    ) -> List[Tuple[int, int, float, str]]:
        # Merge consecutive VAD segments that are closer than voice_enrollment_vad_merge_secs
        merged_vad_data = []
        for segment in vad_data:
            if merged_vad_data and segment["start"] - merged_vad_data[-1]["end"] < self.voice_enrollment_vad_merge_secs:
                merged_vad_data[-1]["end"] = max(merged_vad_data[-1]["end"], segment["end"])
            else:
                merged_vad_data.append(segment)

        # split VAD segments that are longer than max_voice_enrollment_secs into smaller segments
        final_vad_data = []
        for segment in merged_vad_data:
            start = segment["start"]
            end = segment["end"]
            while end - start > self.max_voice_enrollment_secs:
                final_vad_data.append({"start": start, "end": start + self.max_voice_enrollment_secs})
                start += self.max_voice_enrollment_secs
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
        speech_ranges = [
            (start_code, end_code, length_secs, target_channel_chars[start_code:end_code]) 
            for start_code, end_code, length_secs in speech_ranges
        ]

        # sort by duration descending
        speech_ranges.sort(key=lambda x: x[2], reverse=True)

        return speech_ranges

    def _select_target_voice(
        self,
        target_channel_speech_ranges: List[Tuple[int, int, float, str]],
        example_start_code: int,
        example_end_code: int,
        target_min_candidates: int = 20,
        target_min_length_secs: float = 3.0,
    ) -> Optional[Tuple[int, int, str]]:
        # only sample from speech segments outside the current example range.
        target_channel_speech_ranges = [
            (start_code, end_code, length_secs, voice_str) for start_code, end_code, length_secs, voice_str in target_channel_speech_ranges 
            if (end_code <= example_start_code or start_code >= example_end_code) # outside the current example range
        ]
        # take target_min_candidates longest candidates or all that are target_min_length_secs and longer, whichever yields more candidates.
        voice_candidates = [
            (start_code, end_code, voice_str) for i, (start_code, end_code, length_secs, voice_str) in enumerate(target_channel_speech_ranges) 
            if i < target_min_candidates or length_secs >= target_min_length_secs
        ]
        # Select a random voice candidate from voice_candidates
        if not voice_candidates:
            return None
        selected_voice = random.choice(voice_candidates)
        return selected_voice

    def iterate_examples(
        self, 
        codes_path: str,
        vads_path: str,
        codes_filter: Optional[Union[str, List[str]]] = None,
        codes_filter_exclude: Optional[Union[str, List[str]]] = None,
    ) -> Iterator[str]:
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
            num_channels = len(file_channels)
            context_codes = int(self.context_secs * self.codec_framerate * self.num_codebooks * num_channels)
            overlap_codes = int(self.overlap_secs * self.codec_framerate * self.num_codebooks * num_channels)
            if context_codes % (self.num_codebooks * num_channels) != 0 or overlap_codes % (self.num_codebooks * num_channels) != 0:
                raise ValueError(
                    f"context_codes and overlap_codes must be divisible by {self.num_codebooks * num_channels} "
                    "To ensure examples do not start or end in the middle of an acoustic unit or channel triple."
                )
            
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

            # build the codes strings (audio unicode characters in channel-interleaved order)
            codes_strs = []
            if "one_vs_all" in file_root.lower():
                # This is a "one-vs-all" sample where channel 0 has the target speaker and channel 1 has multiple mixed other speakers.
                # We'll generate one example with channel 0 as the target speaker.
                codes_strs.append(self._build_codes_str(channels_chars))
            else:
                # This is a "one-vs-one" sample where each channel has a single speaker.
                # We'll generate two examples, one for each channel as the target speaker.
                codes_strs.append(self._build_codes_str(channels_chars))
                codes_strs.append(self._build_codes_str(channels_chars[:1] + channels_chars[1:][::-1]))

            # build the examples
            random.seed(self.voice_enrollment_selection_seed)
            for target_channel, codes_str in enumerate(codes_strs):
                # load the speech ranges for the target channel for use in selecting target voices for each example
                target_channel_vad_file = os.path.join(vads_path, f"{file_root}_c{target_channel}.json")
                target_channel_vad = self._load_vad(target_channel_vad_file)
                target_channel_chars = channels_chars[1 + target_channel]
                target_channel_speech_ranges = self._preprocess_vad_for_target_voice_selection(target_channel_vad, target_channel_chars)
                if not target_channel_speech_ranges:
                    print(f"No speech ranges found for target channel {target_channel} in file {file_root}. Skipping channel...")
                    continue

                # yield examples from the sequence with the specified sequence length and overlap
                start_code = 0
                example_index = 0
                while True:
                    # slice codes from the full sequence for the current example
                    end_code = min(start_code + context_codes, len(codes_str))
                    example = codes_str[start_code:end_code]

                    # select the target voice for the current example
                    start_code_target = int(start_code / num_channels)
                    end_code_target = int(end_code / num_channels)
                    target_voice = self._select_target_voice(
                        target_channel_speech_ranges,
                        start_code_target,
                        end_code_target,
                    )

                    # yield the example and metadata
                    if target_voice is not None:
                        tv_start_code, tv_end_code, tv_str = target_voice
                        example = f"{self.header_target_voice_token}{tv_str}{self.header_end_token}{example}"
                        metadata = {
                            "file_id": file_root,
                            "target_channel": target_channel,
                            "example_index": example_index,
                            "ex_start_secs": start_code / (self.codec_framerate * self.num_codebooks * num_channels),
                            "ex_end_secs": end_code / (self.codec_framerate * self.num_codebooks * num_channels),
                            "tv_start_secs": tv_start_code / (self.codec_framerate * self.num_codebooks),
                            "tv_end_secs": tv_end_code / (self.codec_framerate * self.num_codebooks),
                        }
                        yield example, metadata
                        example_index += 1
                    else:
                        print(
                            f"Could not find target voice sample outside of range {start_code_target}-{end_code_target} "
                            f"on channel {target_channel} in file {file_root}. Skipping example..."
                        )

                    # move to next start_code, or break if we've reached the end of the sequence
                    if end_code >= len(codes_str):
                        break
                    start_code = end_code - overlap_codes