import numpy as np
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

from . import RealtimeSeparatorResources, RealtimeSeparatorConfig
from .realtime_separator_profiler import RealtimeSeparatorProfilerCollection
from .utils.audio_utils import smooth_join, create_crossfade_ramps, pad_or_trim

class RealtimeSeparator:
    def __init__(self, resources: RealtimeSeparatorResources = None, config: RealtimeSeparatorConfig = None):
        if resources is None:
            resources = RealtimeSeparatorResources()
        self.resources = resources

        if config is None:
            config = RealtimeSeparatorConfig()
        self.set_config(config)

        self.reset()

    @property
    def total_frames(self) -> int:
        return len(self.audio_tokens_idx)

    @property
    def total_secs(self) -> float:
        return self.total_frames / (self.resources.mono_audio_tokenizer.framerate * 3)

    def reset(self):
        self.resources.mono_audio_tokenizer.reset_context()
        self.resources.stereo_audio_tokenizer.reset_context()
        c = self.config

        self.set_sampler()
        self.resources.llm.reset()
        
        voice_enrollment = np.zeros(self.resources.mono_audio_tokenizer.sampling_rate * 10, dtype=np.float32) \
            if c.target_voice_enrollment is None else c.target_voice_enrollment
        enrollment_audio_str = self.resources.mono_audio_tokenizer.chunked_tokenize_audio(voice_enrollment, c.chunk_size_secs)

        separator_prompt = "".join([
            c.header_delay_token.format(delay_ms=int(c.delay_secs*1000)),
            c.header_target_voice_token,
            enrollment_audio_str,
            c.header_end_token,
        ])
        self.input_ids = self.resources.tokenizer.encode(separator_prompt)
        self.context_start_pos = len(self.input_ids)
        self.resources.llm.eval(self.input_ids[:-1])

        self.trim_to_secs = 0.0
        self.audio_history_mono: List[np.ndarray] = [] 
        self.audio_history_stereo: List[np.ndarray] = []
        self.audio_tokens_idx = []

        self.profilers.reset()

    def set_config(self, config: RealtimeSeparatorConfig):
        self.config = config
        
        self.chunk_size_samples = int(self.config.chunk_size_secs * self.resources.mono_audio_tokenizer.sampling_rate)
        self.crossfade_ramps = create_crossfade_ramps(self.resources.stereo_audio_tokenizer.sampling_rate, fade_secs=self.config.chunk_fade_secs)

        self.skip_char_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.skip_char)

        self.resources.stereo_audio_tokenizer.reset_context()
        self.delay_silence = self.resources.stereo_audio_tokenizer.tokenize_audio(np.zeros((2, self.chunk_size_samples), dtype=np.float32))

        self.profilers = RealtimeSeparatorProfilerCollection(self.config)

    def set_sampler(self) -> None:
        c = self.config
        self.resources.llm.init_sampler_for_generate(
            top_k=c.top_k,
            top_p=c.top_p,
            min_p=c.min_p,
            temp=c.temperature,
            repeat_penalty=c.repeat_penalty,
            presence_penalty=c.presence_penalty,
            frequency_penalty=c.frequency_penalty,
            seed=c.seed,
        )

    def trim_sequences(self) -> None:
        if self.total_secs - self.trim_to_secs >= self.config.max_context_secs:
            self.trim_to_secs += self.config.trim_by_secs
            self.recompute_kv_cache(0)

    def process_audio_input_ids(
        self, 
        audio_chunk_input_ids: List[int], 
    ) -> List[int]:
        out_chunk_input_ids = [0] * len(audio_chunk_input_ids) * 2
        for i in range(len(audio_chunk_input_ids)):
            # trim the sequences to the maximum length
            self.trim_sequences()
            # append the next input (user) audio token
            self.input_ids.append(audio_chunk_input_ids[i])
            # predict next two tokens
            if self.total_secs < self.config.delay_secs:
                next_token = next_token_2 = self.skip_char_token_id
                self.resources.llm.eval(self.input_ids[-2:] + [next_token])
            else:
                next_token = next(self.resources.llm.generate(self.input_ids[-2:], reset=False))
                next_token_2 = next(self.resources.llm.generate([next_token], reset=False))
            self.input_ids.extend([next_token, next_token_2])
            self.audio_tokens_idx.extend([len(self.input_ids)-3, len(self.input_ids)-2, len(self.input_ids)-1])
            out_chunk_input_ids[i*2] = next_token
            out_chunk_input_ids[i*2+1] = next_token_2
        return out_chunk_input_ids

    def process_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        with self.profilers.total_profiler:
            # Sanity check - input size
            assert audio_chunk.shape[-1] == self.chunk_size_samples, \
                f"audio_chunk must have length {self.chunk_size_samples}, but got {audio_chunk.shape[-1]}"

            # Encode audio chunk to input ids
            with self.profilers.audio_tokenize_profiler:
                audio_chunk_str = self.resources.mono_audio_tokenizer.tokenize_audio(audio_chunk)
            with self.profilers.tokenize_profiler:
                audio_chunk_input_ids = self.resources.tokenizer.encode(audio_chunk_str, add_special_tokens=False)
            
            # Generate next audio chunk (interleaved by frame with the input audio chunk)
            with self.profilers.lm_profiler:
                out_chunk_input_ids = self.process_audio_input_ids(audio_chunk_input_ids)

            # Decode input ids to audio chunk and append to the audio history
            out_chunk = self.detokenize_output_chunk(out_chunk_input_ids)
            self.audio_history_mono.append(audio_chunk)

            # Sanity check - output size
            assert out_chunk.shape[-1] == self.chunk_size_samples, \
                f"out_chunk must have length {self.chunk_size_samples}, but got {out_chunk.shape[-1]}"
            return out_chunk

    def detokenize_output_chunk(self, out_chunk_input_ids: List[int]) -> np.ndarray:
        with self.profilers.detokenize_profiler:
            out_chunk_str = self.resources.tokenizer.decode(out_chunk_input_ids, skip_special_tokens=False)
            num_skip_chars = out_chunk_str.rfind("-") + 1
            out_chunk_str = self.delay_silence[:num_skip_chars] + out_chunk_str[num_skip_chars:]
        with self.profilers.audio_detokenize_profiler:
            (_, out_chunk), _, preroll_samples = self.resources.stereo_audio_tokenizer.detokenize_audio(out_chunk_str, preroll_samples=self.crossfade_ramps[0])
        out_chunk = pad_or_trim(out_chunk, self.chunk_size_samples + preroll_samples)
        if len(self.audio_history_stereo) > 0:
            joined_stereo_chunks = smooth_join(self.audio_history_stereo[-1], out_chunk, *self.crossfade_ramps)
            # sanity check - joined size
            assert joined_stereo_chunks.shape[-1] == 2 * self.chunk_size_samples, \
                f"joined_stereo_chunks must have length {2 * self.chunk_size_samples}, but got {joined_stereo_chunks.shape[-1]}"
            self.audio_history_stereo[-1] = joined_stereo_chunks[..., :self.chunk_size_samples]
            self.audio_history_stereo.append(joined_stereo_chunks[..., self.chunk_size_samples:])
            # Emit the output chunk from the audio history, shifted left by chunk_fade_secs.
            # This is done because the smooth join (crossfade) modifies the tail of the previous chunk.
            # So, we include the tail of the previous chunk in the output and exclude the tail of the current chunk.
            out_chunk = joined_stereo_chunks[..., -self.chunk_size_samples-self.crossfade_ramps[0]:-self.crossfade_ramps[0]]
        else:
            self.audio_history_stereo.append(out_chunk)
            # In case of the first chunk there will be no previous chunk tail so pad it on the left with zeros
            out_chunk = pad_or_trim(out_chunk[..., :-self.crossfade_ramps[0]], self.chunk_size_samples, pad_side="left")
        return out_chunk

    def frames_from_secs(self, secs: float) -> int:
        frames = int(secs * self.resources.mono_audio_tokenizer.framerate * 3)
        # make sure we return the position at the beginning of an audio token triple
        if frames % 3 != 0:
            frames -= frames % 3
        return frames
        
    def recompute_kv_cache(self, edit_start_pos: int, edit_end_pos: Optional[int] = None) -> None:
        trim_to_frames = self.frames_from_secs(self.trim_to_secs)
        trim_to_pos = self.audio_tokens_idx[trim_to_frames]
        if trim_to_frames == 0 or edit_end_pos is None or edit_end_pos > trim_to_pos:
            start_pos = edit_start_pos if trim_to_frames == 0 else max(edit_start_pos, trim_to_pos)
            self.resources.llm.n_tokens = start_pos if trim_to_frames == 0 else start_pos-trim_to_pos+self.context_start_pos
            eval_tokens = self.input_ids[start_pos:-1]
            if self.config.delay_secs > 0.0:
                eval_tokens = np.array(eval_tokens)
                delay_frames = self.frames_from_secs(self.config.delay_secs)
                eval_tokens[1:delay_frames:3] = self.skip_char_token_id
                eval_tokens[2:delay_frames:3] = self.skip_char_token_id
                eval_tokens = eval_tokens.tolist()
            self.resources.llm.eval(eval_tokens)

    def get_sequence_str(self) -> str:
        separator_sequence = self.resources.tokenizer.decode(self.input_ids, skip_special_tokens=False)
        return separator_sequence
    
    def get_mono_audio_history(self) -> np.ndarray:
        if len(self.audio_history_mono) == 0:
            return np.zeros(0, dtype=np.float32)
        mono_history = np.concatenate(self.audio_history_mono)
        return mono_history

    def get_stereo_audio_history(self) -> np.ndarray:
        if len(self.audio_history_stereo) == 0:
            return np.zeros((2, 0), dtype=np.float32)
        stereo_history = np.concatenate(self.audio_history_stereo, axis=-1)
        return stereo_history  

@dataclass
class RealtimeSeparatorMultiprocessingInfo:
    config: RealtimeSeparatorConfig
    sampling_rate: int
    chunk_size_samples: int
    total_secs: float
    sequence: str
    stereo_audio_history: np.ndarray

class RealtimeSeparatorMultiprocessing:
    def __init__(
        self, 
        wait_until_running: bool = True,
        config: RealtimeSeparatorConfig = None,
        gpu_id: Optional[int] = None,
        idle_tol_secs: float = 1.0,
        **resources_kwargs,
    ):
        import multiprocessing as mp
        from ctypes import c_bool
        ctx = mp.get_context("spawn")
        self.config_queue = ctx.SimpleQueue()
        self.info_queue = ctx.SimpleQueue()
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        self.running = ctx.Value(c_bool, False)
        self.set_config_flag = ctx.Value(c_bool, False)
        self.reset_flag = ctx.Value(c_bool, False)
        self.get_info_flag = ctx.Value(c_bool, False)

        self.execute_process = ctx.Process(
            target=self.execute, 
            daemon=True, 
            args=(config, gpu_id, idle_tol_secs),
            kwargs=resources_kwargs,
        )
        self.execute_process.start()

        if wait_until_running:
            self.wait_until_running()

    def wait_until_running(self):
        #TODO: use an Event instead of a loop
        while not self.is_running():
            time.sleep(0.01)

    def is_running(self):
        return self.running.value

    def execute(self, config: RealtimeSeparatorConfig, gpu_id: Optional[int], idle_tol_secs: float, **resources_kwargs):
        if gpu_id is not None:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        separator_resources = RealtimeSeparatorResources(**resources_kwargs)
        separator = RealtimeSeparator(resources=separator_resources, config=config)
        last_input_time = datetime.now()
        is_idle = False

        self.running.value = True
        print(">>> Separator is running! <<<")
        while True:
            try:
                if self.set_config_flag.value:
                    self.reset_flag.value = True
                    config = self.config_queue.get()
                    separator.set_config(config)
                    self.set_config_flag.value = False
                    print(">>> Config updated! <<<")

                if self.reset_flag.value:
                    separator.reset()
                    self._skip_queue(self.input_queue)
                    self.reset_flag.value = False
                    print(">>> Separator reset! <<<")

                if self.get_info_flag.value:
                    info = RealtimeSeparatorMultiprocessingInfo(
                        config=separator.config,
                        sampling_rate=separator.resources.mono_audio_tokenizer.sampling_rate,
                        chunk_size_samples=separator.chunk_size_samples,
                        total_secs=separator.total_secs,
                        sequence=separator.get_sequence_str(),
                        stereo_audio_history=separator.get_stereo_audio_history(),
                    )
                    self.info_queue.put(info)
                    self.get_info_flag.value = False
                    print(">>> Info retrieved! <<<")

                now = datetime.now()
                if not self.input_queue.empty():
                    input_audio = self.input_queue.get()
                    if isinstance(input_audio, np.ndarray):
                        input_audio = (input_audio, None)
                    output_audio = separator.process_audio(*input_audio)
                    realtime_factor = separator.profilers.total_profiler.realtime_factor_values[-1] \
                        if separator.profilers.total_profiler.realtime_factor_values else None
                    self.output_queue.put((output_audio, realtime_factor))
                    if is_idle:
                        print(">>> Separator is no longer idle! <<<")
                    last_input_time = now
                    is_idle = False
                elif not is_idle:
                    secs_since_last_input = (now - last_input_time).total_seconds()
                    if secs_since_last_input >= idle_tol_secs:
                        print(">>> Separator is idle! <<<")
                        is_idle = True
            except Exception as ex:
                #TODO: logging here
                print(ex)
                #raise ex
            
            if is_idle:
                time.sleep(0.05)

    def _skip_queue(self, queue):
        val = None
        while not queue.empty():
            val = queue.get()
        return val

    def reset(self):
        self.reset_flag.value = True
        #TODO: use an Event instead of a loop
        while self.reset_flag.value:
            time.sleep(0.01)

    def set_config_and_reset(self, config):
        self.set_config_flag.value = True
        self.config_queue.put(config)
        #TODO: use an Event instead of a loop
        while self.set_config_flag.value or self.reset_flag.value:
            time.sleep(0.01)

    def get_info(self) -> RealtimeSeparatorMultiprocessingInfo:
        self.get_info_flag.value = True
        return self.info_queue.get()

    def queue_input(self, input):
        self.input_queue.put(input)

    def next_output(self, block: bool = False):
        if not block and self.output_queue.empty():
            return None
        return self.output_queue.get()
    