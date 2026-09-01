from typing import Tuple
import numpy as np

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

    