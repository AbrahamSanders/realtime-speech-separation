import os
from typing import Dict, Optional, Tuple, Union, List

import librosa
import numpy as np

RAW_AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".opus"]

def try_load_original_audio(
    metadata: Dict, 
    raw_audio_dir: str, 
    length_secs: Optional[float] = None, 
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    m = metadata
    for ext in RAW_AUDIO_EXTENSIONS:
        audio_path = os.path.join(raw_audio_dir, f"{m['file_id']}{ext}")
        if not os.path.exists(audio_path):
            continue

        # target voice audio
        tv_length_secs = m["tv_end_secs"] - m["tv_start_secs"]
        tv_length_secs = min(length_secs, tv_length_secs) if length_secs else tv_length_secs
        tv_audio, tv_sr = librosa.load(audio_path, sr=None, mono=False, offset=m["tv_start_secs"], duration=tv_length_secs)
        tv_audio = tv_audio[m["target_channel"]]

        # example audio
        ex_length_secs = m["ex_end_secs"] - m["ex_start_secs"]
        ex_length_secs = min(length_secs, ex_length_secs) if length_secs else ex_length_secs
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

def get_audio_files(
    path: str, 
    filter_include: Optional[Union[str, List[str]]] = None, 
    filter_exclude: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    if isinstance(filter_include, str):
        filter_include = [filter_include]
    filter_include = [f.lower() for f in filter_include] if filter_include else []

    if isinstance(filter_exclude, str):
        filter_exclude = [filter_exclude]
    filter_exclude = [f.lower() for f in filter_exclude] if filter_exclude else []

    result_files = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_path_lower = file_path.lower()
            if not any(file_path_lower.endswith(extension) for extension in RAW_AUDIO_EXTENSIONS):
                continue
            if filter_include and not any([f in file_path_lower for f in filter_include]):
                continue
            if filter_exclude and any([f in file_path_lower for f in filter_exclude]):
                continue
            result_files.append(file_path)
    result_files.sort()
    return result_files