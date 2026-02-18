from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.static_funcs import create_and_save_report
from pytorch_lightning import LightningModule



class KGE_Literal(LightningModule):
    def __init__(self, kge_model, Literal_model, args, literal_dataset):
        super().__init__()
        self.kge_model = kge_model
        self.Literal_model = Literal_model
        self.args = args
        self.literal_dataset = literal_dataset
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    def model_device(self):
        """Check if all models are on same device"""
        if self.kge_model.device != self.Literal_model.device:
            raise ValueError("KGE and Literal models are on different devices.")

    def on_train_start(self):
        """Called when the train begins."""
        self.start_time = datetime.now()

    def forward(self, x):
        return self.kge_model(x)

    def training_step(self, batch,  batch_idx):
        #get batch_data
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            train_X, train_y, train_t = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            train_X, train_y = batch
            train_t = None
        else:
            raise ValueError(f"Unexpected batch format in training_step: {type(batch)}")
        yhat_e = self.kge_model.forward(train_X)  # (batch_size, num_entities)
        ent_loss = self.bce_loss_fn(yhat_e, train_y)
        self.log("ent_loss", ent_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=train_X.size(0))

        # Literal model (if active)
        if (
            self.Literal_model
            and self.current_epoch > self.args.deferred_literal_training_epochs
            and train_t is not None
        ):
            head = train_X[:, 0].long()
            tail = train_t.flatten().long()
            # stacking head and tail together along a new dimension
            entity_ids = torch.cat([head, tail], dim=0)   # merge into [2 * num_triples]
            entity_ids = torch.unique(entity_ids)         # deduplicate
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
            self.log("lit_loss", lit_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=head.size(0))

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
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            val_X, val_y, _ = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            val_X, val_y = batch
        else:
            raise ValueError(f"Unexpected batch format in validation_step: {type(batch)}")
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
        print("KGE model evaluation started.")
        self.trainer.evaluator.eval(
            dataset=self.trainer.entity_dataset,
            trained_model=self.kge_model,
            form_of_labelling="EntityPrediction",
            during_training=False
        )
        create_and_save_report(self)

        
