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
        if self.literal_dataset is not None:
            self.literal_dataset.warm_entity_literal_vocab(self.device)

    def forward(self, x):
        return self.kge_model(x)

    def _entity_embedding_weight(self):
        entity_embeddings = getattr(self.kge_model, "entity_embeddings", None)
        if entity_embeddings is None or not hasattr(entity_embeddings, "weight"):
            return None
        return entity_embeddings.weight

    def _grad_wrt_entity_weights(self, loss: torch.Tensor, retain_graph: bool = True):
        ent_weight = self._entity_embedding_weight()
        if ent_weight is None:
            return None
        grad = torch.autograd.grad(
            outputs=loss,
            inputs=ent_weight,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )[0]
        return grad

    def _compute_dynamic_lit_weight(
        self,
        ent_loss: torch.Tensor,
        lit_loss: torch.Tensor,
        entity_ids: torch.Tensor,
        target_ratio: float = 1.0,
        min_w: float = 1e-4,
        max_w: float = 10.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if entity_ids is None or entity_ids.numel() == 0:
            return torch.tensor(1.0, device=self.device)

        tracked_ids = torch.unique(entity_ids.long()).to(self.device)
        ent_grad_full = self._grad_wrt_entity_weights(ent_loss, retain_graph=True)
        lit_grad_full = self._grad_wrt_entity_weights(lit_loss, retain_graph=True)
        if ent_grad_full is None:
            return torch.tensor(1.0, device=self.device)

        ent_grad = ent_grad_full.index_select(0, tracked_ids)
        if lit_grad_full is None:
            lit_grad = torch.zeros_like(ent_grad)
        else:
            lit_grad = lit_grad_full.index_select(0, tracked_ids)

        ent_norm = torch.linalg.vector_norm(ent_grad)
        lit_norm = torch.linalg.vector_norm(lit_grad)
        lambda_lit = target_ratio * (ent_norm / (lit_norm + eps))
        return torch.clamp(lambda_lit, min=min_w, max=max_w).detach()

    def training_step(self, batch, batch_idx):
        #get batch_data 
        train_X, train_y = batch

        
        yhat_e = self.kge_model.forward_k_vs_all(train_X)  # (batch_size, num_entities)
        ent_loss = self.bce_loss_fn(yhat_e, train_y)
        self.log("ent_loss", ent_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=train_X.size(0))

        # Literal model (if active)
        if self.Literal_model and self.current_epoch > self.args.deferred_literal_training_epochs:
            head = train_X[:, 0].long()
            entity_ids = torch.unique(head)
            entity_ids = entity_ids.to(self.device)
            lit_entities, lit_properties, y_true = self.literal_dataset.get_batch(entity_ids)
            if y_true.numel() == 0:
                return ent_loss
            ent_embeds = self.kge_model.entity_embeddings(lit_entities)
            yhat_lit = self.Literal_model(ent_embeds, lit_properties)
            lit_loss = F.l1_loss(yhat_lit, y_true)
            self.log("lit_loss", lit_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=head.size(0))
            lambda_lit = self._compute_dynamic_lit_weight(
                ent_loss=ent_loss,
                lit_loss=lit_loss,
                entity_ids=entity_ids,
                target_ratio=float(getattr(self.args, "target_grad_ratio", 1.0)),
                min_w=float(getattr(self.args, "min_lit_weight", 1e-4)),
                max_w=float(getattr(self.args, "max_lit_weight", 10.0)),
                eps=1e-12,
            )
            self.log("lambda_lit", lambda_lit, on_step=True, on_epoch=False, prog_bar=False)
            return ent_loss + lambda_lit * lit_loss

        else:
            return ent_loss

    def validation_step(self, batch):
        val_X, val_y = batch
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

        
