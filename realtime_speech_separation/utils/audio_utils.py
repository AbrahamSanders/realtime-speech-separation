from typing import Tuple, Union
from pydub import AudioSegment
import librosa
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def smooth_join(chunk1: np.ndarray, chunk2: np.ndarray, L: int, fade_in: np.ndarray, fade_out: np.ndarray) -> np.ndarray:
    if chunk1.shape[-1] == 0:
        return chunk2
    if L == 0:
        return np.concatenate((chunk1, chunk2), axis=-1)

    # split tails/heads
    head1, tail1 = chunk1[..., :-L], chunk1[..., -L:]
    head2, tail2 = chunk2[..., :L], chunk2[..., L:]

    # apply ramps
    cross = tail1 * fade_out + head2 * fade_in

    return np.concatenate((head1, cross, tail2), axis=-1)

def create_crossfade_ramps(sr: int, fade_secs: float) -> Tuple[int, np.ndarray, np.ndarray]:
    L = int(sr * fade_secs)
    fade_in = np.sin(0.5 * np.pi * np.linspace(0, 1, L, endpoint=False, dtype=np.float32))
    fade_out = fade_in[::-1]
    return L, fade_in, fade_out

def pad_or_trim(chunk: np.ndarray, target_length: int, pad_side: str = "right") -> np.ndarray:
    if chunk.ndim > 2:
        raise ValueError("Input chunk must be a 1D or 2D array.")
    if chunk.shape[-1] < target_length:
        # Pad with zeros at the end
        pad_width = target_length-chunk.shape[-1]
        pad_width = (0, pad_width) if pad_side == "right" else (pad_width, 0)
        if chunk.ndim == 2:
            pad_width = ((0, 0), pad_width)
        return np.pad(chunk, pad_width, mode="constant")
    elif chunk.shape[-1] > target_length:
        # Trim to target length
        return chunk[..., :target_length]
    else:
        return chunk

def to_resampled_mono(
    input_audio: Tuple[int, np.ndarray], 
    target_sr: int, 
    return_tuple: bool = False,
) -> Union[Tuple[int, np.ndarray], np.ndarray]:
    sr, input_audio = input_audio
    if input_audio.dtype == np.int16:
        input_audio = input_audio.astype("float32") / 32768.0
    if input_audio.ndim == 2:
        if input_audio.shape[1] == 2 and input_audio.shape[0] > 2:
            input_audio = input_audio.T
        input_audio = librosa.to_mono(input_audio) if input_audio.shape[0] > 1 else input_audio[0]
    if sr != target_sr:
        input_audio = librosa.resample(input_audio, orig_sr=sr, target_sr=target_sr)
    if return_tuple:
        return target_sr, input_audio
    return input_audio

def preprocess_enrollment_audio(
    enrollment_audio: Tuple[int, np.ndarray], 
    target_sr: int, 
    min_length_secs: float = 2.0,
    max_length_secs: float = 10.0,
    return_tuple: bool = False,
) -> Union[Tuple[int, np.ndarray], np.ndarray]:
    if enrollment_audio is None:
        return None
    sr, enroll_audio = to_resampled_mono(enrollment_audio, target_sr, return_tuple=True)
    if enroll_audio.shape[-1] > int(max_length_secs * sr):
        enroll_audio = enroll_audio[:int(max_length_secs * sr)]
        logger.info(f"Trimmed enrollment audio to {max_length_secs} seconds.")
    elif enroll_audio.shape[-1] < int(min_length_secs * sr):
        pad_length = int(min_length_secs * sr) - enroll_audio.shape[-1]
        enroll_audio = np.concatenate([enroll_audio, np.zeros(pad_length, dtype=enroll_audio.dtype)])
        logger.info(f"Padded enrollment audio to {min_length_secs} seconds.")
    if return_tuple:
        return sr, enroll_audio
    return enroll_audio

def save_audio(path: str, audio: np.ndarray, sr: int, mp3_quality: int = 2) -> None:
    if audio.ndim == 1:
        channels = 1
    elif audio.ndim == 2:
        channels = audio.shape[0]
        audio = audio.T  # (channels, samples) -> (samples, channels)
    else:
        raise ValueError("Expected mono or multichannel audio")

    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    segment = AudioSegment(
        data=pcm.tobytes(),
        sample_width=2,
        frame_rate=sr,
        channels=channels,
    )

    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp3":
        segment.export(
            path,
            format="mp3",
            codec="libmp3lame",
            parameters=["-q:a", str(mp3_quality)],
        )
    elif ext == ".wav":
        segment.export(
            path,
            format="wav",
            codec="pcm_s16le",
        )
    else:
        raise ValueError(f"Unsupported format: {ext}")