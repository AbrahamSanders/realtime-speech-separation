import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

class DataCollatorWithLossMasking(DataCollatorWithPadding):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, end_header_token: str, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.special_token_ids = torch.tensor([i for i, t in tokenizer.added_tokens_decoder.items() if t.special])
        self.end_header_token_id = tokenizer.convert_tokens_to_ids(end_header_token)

    def __call__(self, features):
        batch = super().__call__(features)
        batch["labels"] = batch["input_ids"].clone()
        labels = batch["labels"]
        input_ids = batch["input_ids"]

        # Mask the loss for the following tokens (set label to -100 to ignore in loss computation):
        # - All tokens before and including the <|end_header|> token (control tokens & voice enrollment audio codes)
        # - All remaining special tokens (padding, skip tokens)
        # - All the M tokens from the audio code triples after the <|end_header|> token e.g. M1 L1 R1 M2 L2 R2 ...
        is_end_header = input_ids == self.end_header_token_id
        body_start = is_end_header.byte().argmax(dim=1) + 1  # first occurrence per row
        body_start_offset = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0) - body_start.unsqueeze(1)

        header_mask = body_start_offset < 0
        special_token_mask = torch.isin(input_ids, self.special_token_ids)
        m_token_in_body_mask = (body_start_offset >= 0) & (body_start_offset % 3 == 0)

        final_token_mask = header_mask | special_token_mask | m_token_in_body_mask
        labels[final_token_mask] = -100

        return batch
