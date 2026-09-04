import json
import os
import librosa
from collections import OrderedDict
from glob import glob
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import streamlit as st

from realtime_speech_separation.audio_tokenizer import AudioTokenizer
from realtime_speech_separation.utils.audio_utils import smooth_join, create_crossfade_ramps

DEFAULT_OUTPUT_DIR = "output"
METADATA_SUFFIX = "_metadata.jsonl"
DEFAULT_RAW_AUDIO_DIR = "data/audio/raw"
RAW_AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".opus"]

@st.cache_resource()
def get_audio_tokenizers() -> Tuple[AudioTokenizer, AudioTokenizer]:
    mono_tokenizer = AudioTokenizer()
    stereo_tokenizer = AudioTokenizer(codec_model=mono_tokenizer.codec_model, num_channels=2)
    return mono_tokenizer, stereo_tokenizer


@st.cache_resource(show_spinner=False)
def load_metadata(metadata_path: str) -> Tuple[OrderedDict, List[Dict[str, Any]]]:
    """Load the metadata jsonl.

    Returns (index, records) where records holds each line's metadata exactly as it
    appears in the file, and index maps:
        file_id -> target_channel -> [line indices into records / the dataset txt file]
    Each line of the metadata jsonl corresponds 1:1 to the same line of the txt file.
    """
    index: OrderedDict[str, OrderedDict[int, List[int]]] = OrderedDict()
    records: List[Dict[str, Any]] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            channels = index.setdefault(meta["file_id"], OrderedDict())
            channels.setdefault(meta["target_channel"], []).append(len(records))
            records.append(meta)
    return index, records


@st.cache_resource(show_spinner=False)
def load_line_offsets(txt_path: str) -> np.ndarray:
    """Byte offset of the start of each line in the dataset txt file.

    Lets any line be read without scanning the file, which can be tens of GB. The
    offsets are computed once by streaming the file, then saved next to it as
    <dataset>.txt.line_offsets.npy so later runs can just load them back.
    """
    line_offsets_path = f"{txt_path}.line_offsets.npy"
    if os.path.exists(line_offsets_path):
        return np.load(line_offsets_path)

    offsets = [0]
    chunk_size = 1 << 24
    pos = 0
    with open(txt_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            start = 0
            while True:
                newline_idx = chunk.find(b"\n", start)
                if newline_idx == -1:
                    break
                offsets.append(pos + newline_idx + 1)
                start = newline_idx + 1
            pos += len(chunk)
    if offsets[-1] >= pos:
        # the file ends with a newline, so the last offset is not the start of a real line
        offsets.pop()

    offsets = np.array(offsets, dtype=np.int64)
    try:
        np.save(line_offsets_path, offsets)
    except OSError:
        pass  # a read-only dataset dir just means we recompute on the next cold start
    return offsets


def read_example(txt_path: str, offsets: np.ndarray, line_idx: int) -> str:
    with open(txt_path, "rb") as f:
        f.seek(int(offsets[line_idx]))
        return f.readline().decode("utf-8").rstrip("\n")


def render_audio(
    audio_str: str,
    audio_tokenizer: AudioTokenizer,
    crossfade_ramps: Tuple[np.ndarray, np.ndarray],
    decoding_chunk_size_secs: float,
    length_secs: float
):
    # stream-decode the reconstruction
    chunk_size_frames = int(decoding_chunk_size_secs * audio_tokenizer.framerate * audio_tokenizer.num_channels)
    decode_frames = min(int(length_secs * audio_tokenizer.framerate * audio_tokenizer.num_channels), len(audio_str)) if length_secs > 0 else len(audio_str)
    audio = np.zeros((audio_tokenizer.num_channels, 0), dtype=np.float32)
    for start in range(0, decode_frames, chunk_size_frames):
        end = start + chunk_size_frames
        chunk = audio_str[start:end]
        (_, output_audio), _, _ = audio_tokenizer.detokenize_audio(chunk, preroll_samples=crossfade_ramps[0])
        audio = smooth_join(audio, output_audio.reshape(audio_tokenizer.num_channels, -1), *crossfade_ramps)
    return audio

def try_load_original_audio(metadata: Dict, raw_audio_dir: str, length_secs: float) -> Optional[np.ndarray]:
    m = metadata
    for ext in RAW_AUDIO_EXTENSIONS:
        audio_path = os.path.join(raw_audio_dir, f"{m['file_id']}{ext}")
        if not os.path.exists(audio_path):
            continue

        # target voice audio
        tv_length_secs = m["tv_end_secs"] - m["tv_start_secs"]
        tv_length_secs = min(length_secs, tv_length_secs) if length_secs > 0 else tv_length_secs
        tv_audio, tv_sr = librosa.load(audio_path, sr=None, mono=False, offset=m["tv_start_secs"], duration=tv_length_secs)
        tv_audio = tv_audio[m["target_channel"]]

        # example audio
        ex_length_secs = m["ex_end_secs"] - m["ex_start_secs"]
        ex_length_secs = min(length_secs, ex_length_secs) if length_secs > 0 else ex_length_secs
        ex_audio_mono, ex_sr_mono = librosa.load(audio_path, sr=None, mono=True, offset=m["ex_start_secs"], duration=ex_length_secs)
        ex_audio_stereo, ex_sr_stereo = librosa.load(audio_path, sr=None, mono=False, offset=m["ex_start_secs"], duration=ex_length_secs)
        # make sure target channel is in position 0 in ex_audio_stereo
        if m["target_channel"] != 0:
            ex_audio_stereo = ex_audio_stereo[::-1]

        sr = tv_sr
        if sr != ex_sr_mono or sr != ex_sr_stereo:
            raise ValueError("Sample rates of target voice and example audio do not match.")
        
        return tv_audio, ex_audio_mono, ex_audio_stereo, sr
    return None

def render_example(
    example: str, 
    chunk_size_secs: float,
    length_secs: float,
    metadata: Dict,
    raw_audio_dir: str,
) -> None:
    """Decode and render the media for a single dataset example.
    """
    header_end_token = "<|end_header|>"
    target_voice_token = "<|target_voice|>"
    st.caption(f"{len(example):,} characters")
    # NOTE: rendered with st.code rather than st.text_area on purpose. A keyed
    # text_area keeps its own widget state, so it would keep showing the first
    # example even after navigating to a different one.
    st.code(example, language=None, wrap_lines=True, height=200)
    header = body = target_voice = None
    if header_end_token in example:
        header, body = example.split(header_end_token, 1)
        if target_voice_token in header:
            _, target_voice = header.split(target_voice_token, 1)
    if target_voice is None:
        st.warning("No target voice found in the header.")
        return
    st.caption(
        f"target voice: {len(target_voice):,} chars · "
        f"body: {len(body):,} chars"
    )
    mono_tokenizer, stereo_tokenizer = get_audio_tokenizers()
    crossfade_ramps = create_crossfade_ramps(mono_tokenizer.sampling_rate, fade_secs=0.02)

    st.write("Decoded Audio (target voice / mono / stereo):")

    # render the target voice audio
    target_voice_audio = render_audio(
        audio_str=target_voice,
        audio_tokenizer=mono_tokenizer,
        crossfade_ramps=crossfade_ramps,
        decoding_chunk_size_secs=chunk_size_secs,
        length_secs=length_secs,
    )
    st.audio(target_voice_audio, sample_rate=mono_tokenizer.sampling_rate)

    # render the mono channel
    mono_str = body[0::3]
    mono_audio = render_audio(
        audio_str=mono_str,
        audio_tokenizer=mono_tokenizer,
        crossfade_ramps=crossfade_ramps,
        decoding_chunk_size_secs=chunk_size_secs,
        length_secs=length_secs,
    )
    st.audio(mono_audio, sample_rate=mono_tokenizer.sampling_rate)

    # render the stereo channel
    stereo_str = "".join(a + b for a, b in zip(body[1::3], body[2::3]))
    stereo_audio = render_audio(
        audio_str=stereo_str,
        audio_tokenizer=stereo_tokenizer,
        crossfade_ramps=crossfade_ramps,
        decoding_chunk_size_secs=chunk_size_secs,
        length_secs=length_secs,
    )
    st.audio(stereo_audio, sample_rate=stereo_tokenizer.sampling_rate)

    st.write("Original Audio (target voice / mono / stereo):")

    orig_audio = try_load_original_audio(metadata, raw_audio_dir, length_secs)
    if orig_audio is None:
        st.info("Original audio not found.")
    else:
        tv_audio, ex_audio_mono, ex_audio_stereo, sr = orig_audio
        st.audio(tv_audio, sample_rate=sr)
        st.audio(ex_audio_mono, sample_rate=sr)
        st.audio(ex_audio_stereo, sample_rate=sr)

def format_time_spans(metadata: Dict[str, Any]) -> str:
    """One-line summary of the time spans in the metadata, for whichever are present."""
    spans = [
        ("example", "ex_start_secs", "ex_end_secs"),
        ("target voice", "tv_start_secs", "tv_end_secs"),
    ]
    parts = []
    for label, start_key, end_key in spans:
        start, end = metadata.get(start_key), metadata.get(end_key)
        if start is None or end is None:
            continue
        parts.append(f"**{label}** {start:.2f}\u2013{end:.2f} s ({end - start:.2f} s)")
    return " \u00b7 ".join(parts)


def find_metadata_files(output_dir: str) -> List[str]:
    return sorted(glob(os.path.join(output_dir, f"*{METADATA_SUFFIX}")))


def set_position(file_i: int, channel_i: int, example_i: int) -> None:
    st.session_state.file_i = file_i
    st.session_state.channel_i = channel_i
    st.session_state.example_i = example_i

def main() -> None:
    st.set_page_config(page_title="Dataset Explorer", layout="wide")
    st.title("Dataset Explorer")

    with st.sidebar:
        st.header("Dataset")
        output_dir = st.text_input("Output directory", DEFAULT_OUTPUT_DIR)
        metadata_files = find_metadata_files(output_dir)
        if not metadata_files:
            st.error(f"No `*{METADATA_SUFFIX}` files found in `{output_dir}`.")
            st.stop()
        metadata_path = st.selectbox(
            "Metadata file",
            metadata_files,
            format_func=os.path.basename,
        )
        raw_audio_dir = st.text_input("Raw audio directory", DEFAULT_RAW_AUDIO_DIR)

    txt_path = metadata_path[: -len(METADATA_SUFFIX)] + ".txt"
    if not os.path.exists(txt_path):
        st.error(f"Expected dataset file `{txt_path}` next to the metadata file.")
        st.stop()

    with st.spinner("Loading metadata..."):
        index, records = load_metadata(metadata_path)
    with st.spinner("Indexing dataset lines (first load on a large file can take a while)..."):
        offsets = load_line_offsets(txt_path)

    if len(offsets) != len(records):
        st.warning(
            f"`{os.path.basename(txt_path)}` has {len(offsets):,} lines but the metadata has "
            f"{len(records):,}. They may be out of sync."
        )

    file_ids = list(index.keys())
    if "metadata_path" not in st.session_state or st.session_state.metadata_path != metadata_path:
        st.session_state.metadata_path = metadata_path
        set_position(0, 0, 0)

    file_i = st.session_state.file_i
    file_id = file_ids[file_i]
    channels = list(index[file_id].keys())
    channel_i = min(st.session_state.channel_i, len(channels) - 1)
    target_channel = channels[channel_i]
    example_lines = index[file_id][target_channel]
    example_i = min(st.session_state.example_i, len(example_lines) - 1)
    set_position(file_i, channel_i, example_i)
    line_idx = example_lines[example_i]

    # Seed the jump widgets from the canonical position. Streamlit ignores a keyed
    # widget's index/value argument once the widget has state, so the widgets are
    # driven by session state instead and stay in sync however the position changed
    # (nav buttons, a dataset switch, or clamping). This must happen before they are
    # created below, and the values must already be clamped to the current options.
    st.session_state.jump_file = file_i
    st.session_state.jump_channel = channel_i
    st.session_state.jump_example = example_i

    # navigation, grouped as [prev/next file] | [prev/next channel] | [prev/next example]
    nav_groups = [
        (
            "file",
            (file_i - 1, 0, 0),
            file_i == 0,
            (file_i + 1, 0, 0),
            file_i >= len(file_ids) - 1,
        ),
        (
            # every channel of a file has the same number of examples, so switching
            # channels holds the example index (handy for A/B-ing the same span of
            # audio across target channels). The clamp above is the safety net if a
            # file ever breaks that assumption.
            "channel",
            (file_i, channel_i - 1, example_i),
            channel_i == 0,
            (file_i, channel_i + 1, example_i),
            channel_i >= len(channels) - 1,
        ),
        (
            "example",
            (file_i, channel_i, example_i - 1),
            example_i == 0,
            (file_i, channel_i, example_i + 1),
            example_i >= len(example_lines) - 1,
        ),
    ]
    for group_col, (name, prev_args, prev_disabled, next_args, next_disabled) in zip(
        st.columns(3), nav_groups
    ):
        prev_col, next_col = group_col.columns(2)
        prev_col.button(
            f"\u25c0 Prev {name}",
            use_container_width=True,
            disabled=prev_disabled,
            on_click=set_position,
            args=prev_args,
        )
        next_col.button(
            f"Next {name} \u25b6",
            use_container_width=True,
            disabled=next_disabled,
            on_click=set_position,
            args=next_args,
        )

    # direct jumps
    with st.sidebar:
        st.header("Jump to")
        st.selectbox(
            "File",
            range(len(file_ids)),
            format_func=lambda i: f"[{i}] {file_ids[i]}",
            key="jump_file",
            on_change=lambda: set_position(st.session_state.jump_file, 0, 0),
        )
        st.selectbox(
            "Target channel",
            range(len(channels)),
            format_func=lambda i: str(channels[i]),
            key="jump_channel",
            on_change=lambda: set_position(file_i, st.session_state.jump_channel, example_i),
        )
        st.number_input(
            "Example index",
            min_value=0,
            max_value=len(example_lines) - 1,
            key="jump_example",
            on_change=lambda: set_position(file_i, channel_i, st.session_state.jump_example),
        )

        st.header("Decoding Settings")
        chunk_size_secs = st.slider("Chunk size (seconds)", min_value=0.02, max_value=1.0, value=1.0, step=0.02)
        length_secs = st.slider("Length (seconds): zero for full length", min_value=0, max_value=300, value=0, step=1)

    # current example
    st.subheader(file_id)
    info_cols = st.columns(4)
    info_cols[0].metric("File", f"{file_i} ({file_i + 1} / {len(file_ids)})")
    info_cols[1].metric("Target channel", f"{target_channel} ({channel_i + 1} / {len(channels)})")
    # example_i is the metadata's 0-based example_index, which is what the jump
    # widget edits, so show it as-is alongside the 1-based position (same format
    # as the target channel metric above).
    info_cols[2].metric("Example", f"{example_i} ({example_i + 1} / {len(example_lines)})")
    info_cols[3].metric("Dataset line", f"{line_idx:,}")

    metadata = records[line_idx]
    time_spans = format_time_spans(metadata)
    if time_spans:
        st.caption(time_spans)

    example = read_example(txt_path, offsets, line_idx)
    render_example(
        example,
        chunk_size_secs=chunk_size_secs,
        length_secs=length_secs,
        metadata=metadata,
        raw_audio_dir=raw_audio_dir,
    )            

    with st.expander("Raw metadata"):
        st.json(metadata)


if __name__ == "__main__":
    main()
