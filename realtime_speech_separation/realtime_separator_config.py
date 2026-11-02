import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

SUPPORTED_DELAY_SECS: List[float] = [
    0.0,
    0.1,
    0.2,
    0.4,
    0.8,
    1.6,
    3.2,
    6.4,
]

@dataclass
class RealtimeSeparatorConfig:
    target_voice_enrollment: Optional[Tuple[int, np.ndarray]] = None
    delay_secs: float = 0.8
    temperature: float = 0.8
    top_k: int = 100
    top_p: float = 1.0
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    chunk_size_secs: float = 0.1
    chunk_fade_secs: float = 0.02
    max_context_secs: float = 60.0
    trim_by_secs: float = 15.0
    seed: Optional[int] = 42
    header_delay_token: str = "<|delay_{delay_ms}ms|>"
    header_target_voice_token: str = "<|target_voice|>"
    header_end_token: str = "<|end_header|>"
    skip_char: str = "-"
    run_profilers: bool = True
    profiler_report_interval_secs: float = 2.0

    def __post_init__(self):
        if int(self.chunk_size_secs*100) % 2 != 0:
            raise ValueError("Chunk size must be a multiple of 0.02 seconds.")
        if self.chunk_fade_secs > self.chunk_size_secs:
            raise ValueError("Chunk fade length cannot be longer than the chunk size.")
        if self.delay_secs not in SUPPORTED_DELAY_SECS:
            raise ValueError(f"Unsupported delay_secs value: {self.delay_secs}. Supported values are: {SUPPORTED_DELAY_SECS}")
