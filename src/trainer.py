from datetime import datetime
import torch
import torch.optim as optim
from src.static_funcs import create_and_save_report
from pytorch_lightning import LightningModule


class KGETrainingModule(LightningModule):
    def __init__(self, kge_model, args):
        super().__init__()
        self.kge_model = kge_model
        self.args = args
        self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    def on_train_start(self):
        """Called when the train begins."""
        self.start_time = datetime.now()

    def forward(self, x):
        return self.kge_model(x)

    def training_step(self, batch,  batch_idx):
        train_X, train_y = batch
        yhat_e = self.kge_model.forward(train_X)  # (batch_size, num_entities)
        ent_loss = self.bce_loss_fn(yhat_e, train_y)
        self.log("ent_loss", ent_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=train_X.size(0))
        return ent_loss

    def validation_step(self, batch):
        val_X, val_y = batch
        yhat_val = self.kge_model(val_X)
        val_loss = self.bce_loss_fn(yhat_val, val_y)
        self.log("ent_loss_val", val_loss, on_step=True, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
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

        
