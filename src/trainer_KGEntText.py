import torch
import torch.optim as optim
from pytorch_lightning import LightningModule

from src.KGEntText.model import KGEntTextModel


class KGEntTextTrainer(LightningModule):
    """Lightning wrapper for Gemma-based KGE-conditioned generation."""

    def __init__(self, args, kge_embeddings, tokenizer=None):
        super().__init__()
        self.args = args
        self.kge_embeddings = kge_embeddings
        self.tokenizer = tokenizer
        self.model = KGEntTextModel(
            kge_embeddings=kge_embeddings,
            model_name=getattr(args, "text_model_name", "google/gemma-3-1b-it"),
            torch_dtype=getattr(args, "model_torch_dtype", "auto"),
            lora_r=getattr(args, "lora_r", 8),
            lora_alpha=getattr(args, "lora_alpha", 16),
            lora_dropout=getattr(args, "lora_dropout", 0.05),
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _shared_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            entity_ids=batch["entity_ids"],
            labels=batch.get("labels"),
            attention_mask=batch.get("attention_mask"),
        )
        loss = outputs["loss"]
        perplexity = torch.exp(loss.detach().clamp(max=20))
        return loss, perplexity

    def training_step(self, batch, batch_idx):
        del batch_idx
        loss, perplexity = self._shared_step(batch)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
        )
        self.log(
            "train_perplexity",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["input_ids"].size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        loss, perplexity = self._shared_step(batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
        )
        self.log(
            "val_perplexity",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["input_ids"].size(0),
        )
        return loss

    def test_step(self, batch, batch_idx):
        del batch_idx
        loss, perplexity = self._shared_step(batch)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
        )
        self.log(
            "test_perplexity",
            perplexity,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
        )
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.args, "lr", 1e-4),
            weight_decay=getattr(self.args, "weight_decay", 0.0),
        )
        self.trainer.strategy.zero_grad_kwargs = {"set_to_none": True}
        return optimizer
