python prep_lm_dataset.py \
    --codes_path=data/audio/codes/MagiCodec-50Hz-Base/0.1s_2.0s \
    --save_path=output/dataset_magicodec_131k_8s_DEBUG.txt \
    --header_delay_tokens \
        "<|delay_0ms|>" \
        "<|delay_100ms|>" \
        "<|delay_200ms|>" \
        "<|delay_400ms|>" \
    --num_examples=10000 \
    --context_secs=8.0 \
    --overlap_secs=6.0 \
    --voice_enrollment_max_secs=6.0