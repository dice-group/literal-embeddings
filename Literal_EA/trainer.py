import pytorch_lightning as pl
import torch
from model import *
import json

class LitModel(pl.LightningModule):
    """PyTorch Lightning module for knowledge graph embedding models"""
    def __init__(self, model_name, d, ent_vec_dim, rel_vec_dim, kwargs, lr, label_smoothing, model_mapping,
                 evaluator = None, er_vocab=None, exp_dir=None):
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
            
        loss = self.model.loss(predictions, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure Adam optimizer"""
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
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