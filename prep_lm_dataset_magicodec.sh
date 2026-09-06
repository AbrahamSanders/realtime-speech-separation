python prep_lm_dataset.py \
    --codes_path=data/audio/codes/MagiCodec-50Hz-Base/0.1s_2.0s \
    --save_path=output/dataset_magicodec_131k_40s.txt

python tools/split_lm_dataset.py \
    --dataset_path=output/dataset_magicodec_131k_40s.txt