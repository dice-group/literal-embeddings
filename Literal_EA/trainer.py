import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from model import *
import json

class LitModel(pl.LightningModule):
    """PyTorch Lightning module for knowledge graph embedding models"""
    def __init__(self, model_name, d, ent_vec_dim, rel_vec_dim, kwargs, lr, label_smoothing, model_mapping,
                 evaluator = None, er_vocab=None, exp_dir=None, lr_decay=1.0, literal_model=None, literal_dataset=None):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.model_mapping = model_mapping
        self.model = model_mapping[model_name](d, ent_vec_dim, rel_vec_dim, **kwargs)
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.num_entities = len(d.entities)
        self.er_vocab = er_vocab

        self.scores_dict = dict()
        self.current_split = None
        self.evaluator = evaluator
        self.exp_dir = exp_dir
        self.lr_decay = lr_decay
        self.literal_model = literal_model
        self.literal_dataset = literal_dataset

    def forward(self, e1_idx, r_idx):
        """Forward pass through the model"""
        return self.model.forward(e1_idx, r_idx)

    def training_step(self, batch):
        """Single training step with BCE loss"""
        e1_idx, r_idx, e2_idx = batch
        predictions = self.model.forward(e1_idx, r_idx)
        
        # Create one-hot targets
        targets = torch.zeros(predictions.size(), device=self.device)
        targets[range(e1_idx.size(0)), e2_idx] = 1.0
        
        # Apply label smoothing if specified
        if self.label_smoothing:
            targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
            
        ent_loss = self.model.loss(predictions, targets)
        if self.literal_model:
            lit_ents = torch.cat([e1_idx, e2_idx], dim=0)
            e_ent, attr, y_true = self.literal_dataset.get_batch(lit_ents)
            e_emb = self.model.E(e_ent)
            y_pred = self.literal_model(e_emb, attr)
            lit_loss = F.mse_loss(y_pred.squeeze(), y_true.float())
            # Combined loss
            scale = torch.log1p(
                2 * ((ent_loss * lit_loss) / (ent_loss + lit_loss))
            ).detach()
            total_loss = (1 - scale) * ent_loss + scale * lit_loss
        else:
            total_loss = ent_loss

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        """Configure Adam optimizer"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def on_fit_end(self):
        """Evaluate model and save results after training"""
        print("Evaluation Report:")
        eval_report = self.evaluator.evaluate(self.model, eval_mode='train_test_val')
        
        # Save evaluation report
        eval_filePath = f"{self.exp_dir}/eval_report.json"
        with open(eval_filePath, 'w') as f:
            json.dump(eval_report, f, indent=4)
        print(f"Evaluation report saved to {eval_filePath}")

        # Save model state
        model_path = f"{self.exp_dir}/model.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")