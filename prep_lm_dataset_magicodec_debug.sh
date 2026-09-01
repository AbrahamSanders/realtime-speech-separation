python prep_lm_dataset.py \
    --codes_path=data/audio/codes/MagiCodec-50Hz-Base/0.1s_2.0s \
    --save_path=output/dataset_magicodec_131k_8s_DEBUG.txt \
    --num_examples=1000 \
    --context_secs=8.0 \
    --overlap_secs=2.0 \
    --max_voice_enrollment_secs=6.0