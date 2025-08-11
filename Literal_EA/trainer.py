import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from model import *
import json
from pytorch_lightning.callbacks import EarlyStopping

class LitModel(pl.LightningModule):
    """PyTorch Lightning module for knowledge graph embedding models"""
    def __init__(self, args, train_dataset, kwargs, model_mapping, evaluator=None, er_vocab=None, 
                 literal_model=None, literal_dataset=None):
        super().__init__()
        # self.save_hyperparameters()  # Disabled: user saves models manually
        self.args = args
        self.model_name = args.model
        self.model_mapping = model_mapping
        self.model = model_mapping[args.model](train_dataset, args.edim, args.rdim, **kwargs)
        self.lr = args.lr
        self.label_smoothing = args.label_smoothing
        self.num_entities = len(train_dataset.entities)
        self.er_vocab = er_vocab

        self.scores_dict = dict()
        self.current_split = None
        self.evaluator = evaluator
        self.exp_dir = args.exp_dir
        self.lr_decay = 1.0
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
            if self.args.dynamic_weighting:
                # Dynamic weighting based on losses
                scale = torch.log1p(
                    2 * ((ent_loss * lit_loss) / (ent_loss + lit_loss))
                ).detach()
                # Bound scale between, for example, 0.1 and 0.9
                scale = torch.clamp(scale, min=0.0001, max=0.9999999)
                total_loss = (1 - scale) * ent_loss + scale * lit_loss
            else:
                # Static weighting
                total_loss = self.args.w1 * ent_loss + self.args.w2 * lit_loss
        else:
            total_loss = ent_loss

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def on_validation_epoch_end(self):
        """Compute validation MRR using evaluator for early stopping"""
        if self.evaluator is None:
            return
            
        # Use evaluator to get validation MRR
        eval_results = self.evaluator.evaluate(self.model, eval_mode='val')
        
        if eval_results and 'val' in eval_results:
            val_mrr = eval_results['val'].get('MRR', 0.0)
            self.log("val_mrr", val_mrr, prog_bar=True, sync_dist=True)
            print(f"Validation MRR: {val_mrr:.4f}")

    def configure_optimizers(self):
        """Configure Adam optimizer"""
        if self.literal_model:
            # Use different learning rate for literal model
            optimizer = torch.optim.Adam([
                {'params': self.model.parameters(), 'lr': self.lr},
                {'params': self.literal_model.parameters(), 'lr': self.lr }
            ])
        else:
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

        # Save configuration
        config_to_save = vars(self.args)
        with open(os.path.join(self.args.exp_dir, "config.json"), "w") as f:
            json.dump(config_to_save, f, indent=4)
