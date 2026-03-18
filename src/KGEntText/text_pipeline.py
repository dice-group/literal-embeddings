import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


PROMPT_TEMPLATE = (
    "{entity_name}: "
)


def load_text_tokenizer(tokenizer_name="google/gemma-3-1b-it"):
    """Load a pretrained tokenizer for KGEntText."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_prompt(entity_name):
    return PROMPT_TEMPLATE.format(entity_name=entity_name)


class KGEntTextCollator:
    """Convert raw dataset samples into prompted causal-LM tensors."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _encode_sample(self, sample):
        prompt = build_prompt(sample["entity_name"])
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        description_ids = self.tokenizer.encode(
            sample["description"],
            add_special_tokens=False,
        )

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            description_ids = description_ids + [eos_id]

        token_ids = (prompt_ids + description_ids)[: self.max_length]
        prompt_len = min(len(prompt_ids), len(token_ids))
        labels = [-100] * prompt_len + token_ids[prompt_len:]
        labels = labels[: self.max_length]

        attention_mask = [1] * len(token_ids)
        return token_ids, labels, attention_mask

    def __call__(self, batch):
        encoded_batch = [self._encode_sample(sample) for sample in batch]
        batch_size = len(encoded_batch)

        input_ids = torch.full(
            (batch_size, self.max_length),
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, self.max_length),
            fill_value=-100,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.long)
        entity_ids = torch.tensor(
            [sample["entity_idx"] for sample in batch],
            dtype=torch.long,
        )

        for idx, (token_ids, label_ids, mask_ids) in enumerate(encoded_batch):
            seq_len = len(token_ids)
            input_ids[idx, :seq_len] = torch.tensor(token_ids, dtype=torch.long)
            labels[idx, :seq_len] = torch.tensor(label_ids, dtype=torch.long)
            attention_mask[idx, :seq_len] = torch.tensor(mask_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "entity_ids": entity_ids,
        }


def build_text_dataloader(
    dataset,
    tokenizer,
    batch_size,
    max_length,
    shuffle=True,
    num_workers=0,
):
    collator = KGEntTextCollator(tokenizer=tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
