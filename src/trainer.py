import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule


class KGEModelLightning(LightningModule):
    def __init__(self, kge_model, Literal_model, args, literal_dataset):
        super().__init__()
        self.kge_model = kge_model
        self.Literal_model = Literal_model
        self.args = args
        self.literal_dataset = literal_dataset
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()

        # Define optimizers
        if self.Literal_model:
            self.optimizer = optim.Adam(
                [
                    {"params": self.kge_model.parameters(), "lr": args.lr},
                    {"params": self.Literal_model.parameters(), "lr": args.lit_lr},
                ]
            )
        else:
            self.optimizer = optim.Adam(
                [{"params": self.kge_model.parameters(), "lr": args.lr}]
            )

    def forward(self, x):
        return self.kge_model(x)

    def training_step(self, batch, batch_idx):
        train_X, train_y = batch
        train_X, train_y = train_X.to(self.device), train_y.to(self.device)

        # Always train KGE model
        yhat_e = self.kge_model(train_X)
        ent_loss = self.bce_loss_fn(yhat_e, train_y)
        self.log("ent_loss", ent_loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.Literal_model and  self.current_epoch > self.args.deffered_literal_training_epochs:
            # Literal model forward
            entity_ids = train_X[:, 0].long().to("cpu")
            lit_entities, lit_properties, y_true = self.literal_dataset.get_batch(
                entity_ids,
                multi_regression=self.args.multi_regression,
                random_seed=batch_idx * self.current_epoch,
            )
            lit_entities, lit_properties, y_true = (
                lit_entities.to(self.device),
                lit_properties.to(self.device),
                y_true.to(self.device),
            )

            ent_embeds = self.kge_model.entity_embeddings(lit_entities)
            yhat_lit = self.Literal_model(
                ent_embeds, lit_properties, train_ent_embeds=True
            )
            lit_loss = F.l1_loss(yhat_lit, y_true)
            self.log("lit_loss", lit_loss, on_step=False, on_epoch=True, prog_bar=True)

            # Combined loss
            scale = torch.log1p(
                2 * ((ent_loss * lit_loss) / (ent_loss + lit_loss))
            ).detach()
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
        return self.optimizer

    def on_fit_end(self):
        self.trainer.evaluator.eval(
            dataset=self.trainer.entity_dataset,
            trained_model=self.kge_model,
            form_of_labelling="EntityPrediction",
            during_training=False
        )
