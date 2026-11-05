import argparse
import hashlib
import json
import os

def add_common_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm_model_path", 
        default="Qwen3-1.7B-magicodec-no-bpe-131k-test/Qwen3-1.7B-magicodec-no-bpe-131k-test-F16.gguf", 
        help="Path to the model GGUF file.",
    )

def get_inference_args_hash(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    # get inference group
    inf_group = next((g for g in parser._action_groups if g.title == "Inference Arguments"), None)
    if inf_group is None:
        raise ValueError("Inference Arguments group not found in the parser.")
    inf_args = {a.dest: getattr(args, a.dest) for a in inf_group._group_actions}
    inf_args["llm_model_path"] = args.llm_model_path
    hash_str = hashlib.md5(json.dumps(inf_args, sort_keys=True).encode()).hexdigest()
    hash_str = hash_str[:8]  # shorten the hash for file names

    # If not exists, save a json file with the inference arguments
    json_file = os.path.join(args.input_dir, f"args_{hash_str}.json")
    if not os.path.exists(json_file):
        os.makedirs(args.input_dir, exist_ok=True)
        with open(json_file, "w") as f:
            json.dump(inf_args, f, indent=4)

    return hash_str