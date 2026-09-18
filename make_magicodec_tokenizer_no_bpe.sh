python -m codec_bpe.train_tokenizer \
    --codes_path data/audio/codes/MagiCodec-50Hz-Base/0.1s_2.0s/mono \
    --vocab_size 131084 \
    --pad_token "<|pad|>" \
    --special_tokens \
        "<|pad|>" \
        "<|delay_0ms|>" \
        "<|delay_100ms|>" \
        "<|delay_200ms|>" \
        "<|delay_400ms|>" \
        "<|delay_800ms|>" \
        "<|delay_1600ms|>" \
        "<|delay_3200ms|>" \
        "<|delay_6400ms|>" \
        "<|target_voice|>" \
        "<|end_header|>" \
        "-" \
    --max_token_codebook_ngrams 0 \
    --unicode_offset 0xE000 \
    --save_path output/magicodec_no_bpe_1cb_131k