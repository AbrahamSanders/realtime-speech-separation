import argparse

def add_common_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm_model_path", 
        default="Qwen3-1.7B-magicodec-no-bpe-131k-test/Qwen3-1.7B-magicodec-no-bpe-131k-test-F16.gguf", 
        help="Path to the model GGUF file.",
    )