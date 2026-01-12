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
        train_X, train_y, train_t = batch  # train_t: tail entities for literals

        yhat_e = self.kge_model.forward(train_X)
        loss = self.bce_loss_fn(yhat_e, train_y)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=train_X.size(0))

        return loss

    def validation_step(self, batch):
        val_X, val_y, _ = batch
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

        
