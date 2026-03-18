"""Evaluation helpers for the KGEntText pipeline."""

import json
import os

import torch

from src.KGEntText.text_pipeline import build_prompt

@torch.no_grad()
def generate_description(model, tokenizer, entity_id, entity_name, max_new_tokens=40, device=None):
    """Greedy generation for a single entity-conditioned description."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    prompt = build_prompt(entity_name)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    entity_ids = torch.tensor([entity_id], dtype=torch.long, device=device)

    generated = input_ids
    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated, entity_ids=entity_ids)
        next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break
        if generated.size(1) >= model.max_seq_len:
            break

    completion_ids = generated[0, len(prompt_ids):]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def collect_test_predictions(model, tokenizer, test_dataset, max_examples=None, max_new_tokens=40):
    """Collect reference and generated descriptions for held-out test samples."""
    total_examples = len(test_dataset) if max_examples is None else min(len(test_dataset), max_examples)
    rows = []
    for idx in range(total_examples):
        sample = test_dataset[idx]
        generated = generate_description(
            model=model,
            tokenizer=tokenizer,
            entity_id=sample["entity_idx"],
            entity_name=sample["entity_name"],
            max_new_tokens=max_new_tokens,
        )
        rows.append(
            {
                "entity": sample["entity"],
                "entity_name": sample["entity_name"],
                "reference": sample["description"],
                "generated": generated,
            }
        )
    return rows


def print_test_predictions(model, tokenizer, test_dataset, max_examples=None, max_new_tokens=40):
    """Print reference and generated descriptions for held-out test samples."""
    rows = collect_test_predictions(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        max_examples=max_examples,
        max_new_tokens=max_new_tokens,
    )
    print("\nTest set qualitative predictions:")
    for row in rows:
        print(f"\nEntity: {row['entity_name']} ({row['entity']})")
        print(f"Reference: {row['reference']}")
        print(f"Generated: {row['generated']}")
    return rows


def save_test_predictions(rows, output_path):
    """Persist test predictions as JSON Lines."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
