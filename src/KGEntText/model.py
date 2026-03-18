"""Model definitions for the KGEntText pipeline."""

from __future__ import annotations

import torch
from torch import nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


class KGEntTextModel(nn.Module):
    """Gemma-based causal LM conditioned on a projected KGE prefix token."""

    def __init__(
        self,
        kge_embeddings,
        model_name="google/gemma-3-1b-it",
        torch_dtype="auto",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()
        if kge_embeddings is None:
            raise ValueError("kge_embeddings must be provided.")

        kge_tensor = torch.as_tensor(kge_embeddings, dtype=torch.float32)
        if kge_tensor.ndim != 2:
            raise ValueError("kge_embeddings must have shape [num_entities, kge_dim].")

        self.kge_embeddings = nn.Embedding.from_pretrained(kge_tensor, freeze=False)
        self.num_entities, self.kge_dim = kge_tensor.shape
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.hidden_dim = self.llm.config.hidden_size
        self.max_seq_len = getattr(self.llm.config, "max_position_embeddings", 2048)
        self.kge_projection = nn.Sequential(
            nn.Linear(self.kge_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _prepend_kge_prefix(self, input_ids, entity_ids, attention_mask=None, labels=None):
        token_embeddings = self.llm.get_input_embeddings()(input_ids)
        entity_kge = self.kge_embeddings(entity_ids)
        kge_prefix = self.kge_projection(entity_kge).unsqueeze(1)
        kge_prefix = kge_prefix.to(dtype=token_embeddings.dtype)
        inputs_embeds = torch.cat([kge_prefix, token_embeddings], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prefix_attention = torch.ones(
            (input_ids.size(0), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), 1),
                fill_value=-100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)

        return inputs_embeds, attention_mask, labels

    def forward(self, input_ids, entity_ids, labels=None, attention_mask=None):
        inputs_embeds, attention_mask, labels = self._prepend_kge_prefix(
            input_ids=input_ids,
            entity_ids=entity_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {
            "logits": outputs.logits[:, 1:, :],
            "loss": outputs.loss,
        }
