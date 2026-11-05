python extract_eval_inputs.py \
    --metadata=output/dataset_magicodec_131k_60s_dev_metadata.jsonl \
    --sampling_p=0.05 \
    --sampling_seed=42

python inference_cli.py \
    --input_dir=output/eval/dataset_magicodec_131k_60s_dev_metadata \
    --llm_model_path=Qwen3-1.7B-magicodec-no-bpe-131k-test/Qwen3-1.7B-magicodec-no-bpe-131k-test-F16.gguf \
    --delay_secs=0.8 \
    --seed=42