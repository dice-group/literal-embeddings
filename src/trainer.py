from datetime import datetime
from polars import head
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule

import json


class KGE_Literal(LightningModule):
    def __init__(self, kge_model, Literal_model, args, literal_dataset):
        super().__init__()
        self.kge_model = kge_model
        self.Literal_model = Literal_model
        self.args = args
        self.literal_dataset = literal_dataset
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()

        
    def on_train_start(self):
        """Called when the train begins."""
        self.start_time = datetime.now()

    def forward(self, x):
        return self.kge_model(x)

    def training_step(self, batch, batch_idx):
        # batch: (batch_size, 3) => [:,0]=h, [:,1]=r, [:,2]=t
        train_X = batch[:, :2]    # (batch_size, 2) -> h,r
        train_t = batch[:, 2]     # (batch_size,)   -> t

        # Forward through KGE model
        yhat_e = self.kge_model(train_X)   # (batch_size, num_entities)

        # Create one-hot targets
        targets = torch.zeros_like(yhat_e, device=self.device)
        targets[torch.arange(train_X.size(0), device=self.device), train_t] = 1.0

        # Entity loss
        ent_loss = self.bce_loss_fn(yhat_e, targets)
        self.log("ent_loss", ent_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Literal model (if active)
        if self.Literal_model and self.current_epoch > self.args.deferred_literal_training_epochs:
            head = train_X[:, 0].long()
            tail = train_t.long()

            # Example: stacking head and tail together along a new dimension
            entity_ids = torch.stack([head, tail], dim=1)  # shape [num_triples, 2]
            lit_entities, lit_properties, y_true = self.literal_dataset.get_batch(entity_ids)
            lit_entities, lit_properties, y_true = (
                lit_entities.to(self.device),
                lit_properties.to(self.device),
                y_true.to(self.device),
            )

            ent_embeds = self.kge_model.entity_embeddings(lit_entities)
            # Ensure embeddings are on the same device as the literal model
            ent_embeds = ent_embeds.to(self.device)
            yhat_lit = self.Literal_model(ent_embeds, lit_properties)
            lit_loss = F.l1_loss(yhat_lit, y_true)
            self.log("lit_loss", lit_loss, on_step=False, on_epoch=True, prog_bar=True)

            # Combined loss
            scale = torch.log1p(
                2 * ((ent_loss * lit_loss) / (ent_loss + lit_loss))
            ).detach()
            scale = torch.clamp(scale, min = 1e-9, max= 0.99999)
            total_loss = (1 - scale) * ent_loss + scale * lit_loss
            return total_loss

        else:
            return ent_loss

    def validation_step(self, batch):
        val_X, val_y = batch
        val_X, val_y = val_X.to(self.device), val_y.to(self.device)
        yhat_val = self.kge_model(val_X)
        val_loss = self.bce_loss_fn(yhat_val, val_y)
        self.log("ent_loss_val", val_loss, on_step=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        # Define optimizers
        if self.Literal_model:
            self.optimizer = optim.Adam(
                [
                    {"params": self.kge_model.parameters(), "lr": self.args.lr},
                    {"params": self.Literal_model.parameters(), "lr": self.args.lit_lr},
                ]
            )
        else:
            self.optimizer = optim.Adam(
                [{"params": self.kge_model.parameters(), "lr": self.args.lr}]
            )
        self.trainer.strategy.zero_grad_kwargs = {'set_to_none': True}
        return self.optimizer

    def on_fit_end(self):
        self.kge_model.to("cpu")
        self.kge_model.eval()
        self.trainer.evaluator.eval(
            dataset=self.trainer.entity_dataset,
            trained_model=self.kge_model,
            form_of_labelling="EntityPrediction",
            during_training=False
        )

        num_entities = self.kge_model.num_entities
        num_relations = self.kge_model.num_relations
        train_triples = len(self.trainer.entity_dataset.train_set)
        # Calculate model size and parameters
        total_params = sum(p.numel() for p in self.kge_model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes per param)
        path_experiment = self.args.full_storage_path
        run_time = datetime.now() - self.start_time
        runtime_seconds = run_time.total_seconds()

        # Create the report
        report = {
            "num_train_triples": train_triples,
            "num_entities": num_entities,
            "num_relations": num_relations,
            "max_length_subword_tokens": None,
            "runtime_kg_loading": getattr(self.args, '_kg_loading_time', None),
            "EstimatedSizeMB": model_size_mb,
            "NumParam": total_params,
            "path_experiment_folder": path_experiment,
            "Runtime": runtime_seconds
        }
        report_path = f"{path_experiment}/report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
