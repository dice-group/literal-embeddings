from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule

from src.static_funcs import create_and_save_report
from src.static_train_utils import grad_wrt_entity_weights


class _BaseKGETrainer(LightningModule):
    def __init__(self, kge_model, args, literal_dataset=None):
        super().__init__()
        self.kge_model = kge_model
        self.args = args
        self.literal_dataset = literal_dataset
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    def on_train_start(self):
        self.start_time = datetime.now()
        if self.literal_dataset is not None:
            self.literal_dataset.warm_entity_literal_vocab(self.device)

    def on_train_batch_start(self, batch, batch_idx):
        del batch_idx
        entity_embeddings = getattr(self.kge_model, "entity_embeddings", None)
        if entity_embeddings is None or not hasattr(entity_embeddings, "prepare_batch"):
            return
        train_X, _ = batch
        head_ids = torch.unique(train_X[:, 0].long()).to(self.device)
        entity_embeddings.prepare_batch(head_ids)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        del outputs, batch, batch_idx
        entity_embeddings = getattr(self.kge_model, "entity_embeddings", None)
        if entity_embeddings is not None and hasattr(entity_embeddings, "clear_batch_cache"):
            entity_embeddings.clear_batch_cache()

    def forward(self, x):
        return self.kge_model(x)

    def validation_step(self, batch):
        val_X, val_y = batch
        yhat_val = self.kge_model(val_X)
        val_loss = self.bce_loss_fn(yhat_val, val_y)
        self.log("ent_loss_val", val_loss, on_step=True, on_epoch=True)
        return val_loss

    def on_fit_end(self):
        self.kge_model.to("cpu")
        self.kge_model.eval()
        print("KGE model evaluation started.")
        self.trainer.evaluator.eval(
            dataset=self.trainer.entity_dataset,
            trained_model=self.kge_model,
            form_of_labelling="EntityPrediction",
            during_training=False,
        )
        create_and_save_report(self)


class KGEEntityTrainer(_BaseKGETrainer):
    """Trainer for pure KGE, LiteralE, and KBLN entity-prediction models."""

    def training_step(self, batch, batch_idx):
        del batch_idx
        train_X, train_y = batch
        yhat_e = self.kge_model.forward_k_vs_all(train_X)
        ent_loss = self.bce_loss_fn(yhat_e, train_y)
        self.log("ent_loss", ent_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=train_X.size(0))
        return ent_loss

    def configure_optimizers(self):
        optimizer = optim.Adam([{"params": self.kge_model.parameters(), "lr": self.args.lr}])
        self.trainer.strategy.zero_grad_kwargs = {"set_to_none": True}
        return optimizer


class KGECombinedTrainer(_BaseKGETrainer):
    """Trainer for joint KGE + literal prediction training."""

    def __init__(self, kge_model, literal_model, args, literal_dataset):
        super().__init__(kge_model=kge_model, args=args, literal_dataset=literal_dataset)
        self.Literal_model = literal_model
        self.log_vars = torch.nn.Parameter(torch.zeros(2))

    def training_step(self, batch, batch_idx):
        del batch_idx
        train_X, train_y = batch
        yhat_e = self.kge_model.forward_k_vs_all(train_X)
        ent_loss = self.bce_loss_fn(yhat_e, train_y)
        self.log("ent_loss", ent_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=train_X.size(0))

        if self.current_epoch <= self.args.deferred_literal_training_epochs:
            return ent_loss

        head = train_X[:, 0].long()
        entity_ids = torch.unique(head).to(self.device)
        lit_entities, lit_properties, y_true = self.literal_dataset.get_batch(entity_ids)
        if y_true.numel() == 0:
            return ent_loss

        ent_embeds = self.kge_model.entity_embeddings(lit_entities)
        yhat_lit = self.Literal_model(ent_embeds, lit_properties)
        lit_loss = F.l1_loss(yhat_lit, y_true)
        self.log("lit_loss", lit_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=head.size(0))

        if self.args.combined_loss_strategy == "uncertainty":
            s_ent, s_lit = self.log_vars[0], self.log_vars[1]
            return (
                torch.exp(-s_ent) * ent_loss + s_ent +
                torch.exp(-s_lit) * lit_loss + s_lit
            )

        if self.args.combined_loss_strategy == "harmonic":
            scale = torch.log1p(
                2 * ((ent_loss * lit_loss) / (ent_loss + lit_loss + 1e-12))
            ).detach()
            scale = torch.clamp(scale, min=1e-9, max=0.99999)
            return (1.0 - scale) * ent_loss + scale * lit_loss

        ent_grad = grad_wrt_entity_weights(self.kge_model, ent_loss, retain_graph=True)
        lit_grad = grad_wrt_entity_weights(self.kge_model, lit_loss, retain_graph=True)
        lit_mix = torch.tensor(self.args.lit_weight, device=self.device)
        if ent_grad is not None and lit_grad is not None:
            ent_norm = torch.linalg.vector_norm(ent_grad)
            lit_norm = torch.linalg.vector_norm(lit_grad)
            if torch.isfinite(ent_norm) and torch.isfinite(lit_norm) and lit_norm > 1e-12:
                lit_mix = torch.minimum(lit_mix, ent_norm / (lit_norm + 1e-12)).detach()
        return (1.0 - lit_mix) * ent_loss + lit_mix * lit_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            [
                {"params": self.kge_model.parameters(), "lr": self.args.lr},
                {"params": self.Literal_model.parameters(), "lr": self.args.lit_lr},
                {"params": [self.log_vars], "lr": self.args.lr},
            ]
        )
        self.trainer.strategy.zero_grad_kwargs = {"set_to_none": True}
        return optimizer
