from typing import Tuple
import gradio as gr
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt

from . import RealtimeSeparator, RealtimeSeparatorResources, add_common_inference_args
from .utils.audio_utils import pad_or_trim, to_resampled_mono, preprocess_enrollment_audio
from .realtime_separator_config import SUPPORTED_DELAY_SECS

logger = logging.getLogger(__name__)

separator = None

def set_config_and_reset(
    enrollment_audio: Tuple[int, np.ndarray],
    delay_secs: float,
    seed: int,
    chunk_size_secs: float,
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repeat_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    max_context_secs: float,
    trim_by_secs: float,
    run_profilers: bool,
):    
    config = separator.config
    config.target_voice_enrollment = preprocess_enrollment_audio(
        enrollment_audio, 
        separator.resources.mono_audio_tokenizer.sampling_rate, 
        return_tuple=True,
    )
    config.delay_secs = float(delay_secs)
    config.seed = int(seed) if seed else None
    config.chunk_size_secs = float(chunk_size_secs)
    config.temperature = float(temperature)
    config.top_k = int(top_k)
    config.top_p = float(top_p)
    config.min_p = float(min_p)
    config.repeat_penalty = float(repeat_penalty)
    config.presence_penalty = float(presence_penalty)
    config.frequency_penalty = float(frequency_penalty)
    config.max_context_secs = float(max_context_secs)
    config.trim_by_secs = float(trim_by_secs)
    config.run_profilers = bool(run_profilers)

    separator.set_config(config)
    separator.reset()

def run_separator(
    input_audio: Tuple[int, np.ndarray],
    stream_output: bool,
    *config_args,
):
    set_config_and_reset(*config_args)

    sr, input_audio = to_resampled_mono(input_audio, separator.resources.mono_audio_tokenizer.sampling_rate, return_tuple=True)
    input_audio_length = input_audio.shape[-1]
    # If there is a delay, pad the input audio with silence at the end to make sure we stream the entire output
    if separator.config.delay_secs > 0.0:
        delay_samples = int(separator.config.delay_secs * sr)
        input_audio = np.concatenate([input_audio, np.zeros(delay_samples, dtype=np.float32)])

    # Do the separation and stream the output if requested
    realtime_plot = None
    for start in range(0, input_audio.shape[-1], separator.chunk_size_samples):
        end = start + separator.chunk_size_samples
        chunk = input_audio[start:end]
        chunk = pad_or_trim(chunk, separator.chunk_size_samples)
        _ = separator.process_audio(chunk)

        if end >= input_audio.shape[-1] or end % int(sr * separator.config.profiler_report_interval_secs) == 0:
            if separator.config.run_profilers:
                if realtime_plot is not None:
                    plt.close(realtime_plot)
                realtime_plot = separator.profilers.build_plot()
            if stream_output or end >= input_audio.shape[-1]:
                output_audio = separator.get_stereo_audio_history()
                # If there is a delay, trim the silence at the front of the output audio
                if separator.config.delay_secs > 0.0:
                    output_audio = output_audio[..., delay_samples:]
                # trim any silence from final chunk padding at the end of the output audio
                if end >= input_audio.shape[-1]:
                    output_audio = output_audio[..., :input_audio_length]
                sequence = separator.get_sequence_str()
                yield realtime_plot, (sr, output_audio.T) if output_audio.shape[-1] > 0 else None, sequence
            elif realtime_plot is not None:
                yield realtime_plot, None, None

def main(argv: list[str] | None = None) -> int:
    """Run the Realtime Speech Separation gradio client."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_inference_args(parser)
    
    args = parser.parse_args(argv)
    print(f"Running with args: {args}")
    logging.basicConfig(level=logging.INFO)

    separator = RealtimeSeparator(
        resources=RealtimeSeparatorResources(
            llm_model_path=args.llm_model_path,
        ),
    )

    config = separator.config
    interface = gr.Interface(
        fn=run_separator,
        inputs=[
            gr.Audio(label="Input Audio (Converted to mono if stereo)"),
            gr.Checkbox(False, label="Stream Output"),
            gr.Audio(
                (config.target_voice_enrollment[0], config.target_voice_enrollment[1].T) 
                    if config.target_voice_enrollment is not None and config.target_voice_enrollment[1].ndim == 2 
                    else config.target_voice_enrollment, 
                label="Target Voice Enrollment (Converted to mono if stereo)",
            ),
            gr.Dropdown(choices=SUPPORTED_DELAY_SECS, value=config.delay_secs, label="Delay (seconds)"),
            gr.Number(config.seed, minimum=0, step=1, label="Random seed (0 for random)"),
            gr.Slider(0.02, 1.0, value=config.chunk_size_secs, step=0.02, label="Chunk Size (seconds)"),
            gr.Slider(0.0, 2.0, value=config.temperature, step=0.05, label="Temperature"),
            gr.Slider(0, 500, value=config.top_k, step=1, label="Top-k"),
            gr.Slider(0.0, 1.0, value=config.top_p, step=0.01, label="Top-p"),
            gr.Slider(0.0, 1.0, value=config.min_p, step=0.001, label="Min-p"),
            gr.Slider(0.0, 2.0, value=config.repeat_penalty, step=0.05, label="Repeat Penalty"),
            gr.Slider(-2.0, 2.0, value=config.presence_penalty, step=0.05, label="Presence Penalty"),
            gr.Slider(-2.0, 2.0, value=config.frequency_penalty, step=0.05, label="Frequency Penalty"),
            gr.Number(config.max_context_secs, minimum=5.0, maximum=60.0, step=5.0, label="Max Context Length (seconds)"),
            gr.Number(config.trim_by_secs, minimum=1.0, maximum=15.0, step=1.0, label="Trim By (seconds)"),
            gr.Checkbox(config.run_profilers, label="Run Profilers"),
        ], 
        outputs=[
            gr.Plot(label="Realtime Factor Plot"),
            gr.Audio(label="Audio Output (Separated Stereo)"),
            gr.TextArea(label="Sequence"),
        ],
        allow_flagging='never'
    )
    interface.launch()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
