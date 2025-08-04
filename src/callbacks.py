import pandas as pd
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm import tqdm

from src.dataset import LiteralDataset
from src.model import LiteralEmbeddings


class EpochLevelProgressBar(TQDMProgressBar):
    """Custom progress bar for PyTorch Lightning that displays epoch-level metrics."""
    def __init__(self):
        super().__init__()
        self.epoch_bar = None
        self._train_progress_bar = None

    def on_train_start(self, trainer, pl_module):
        total_epochs = trainer.max_epochs
        self.epoch_bar = tqdm(
            total=total_epochs,
            desc='[Training]',
            leave=False,
            dynamic_ncols=True,
            bar_format='{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt} {postfix}]'
        )
        # # set a dummy disabled bar to avoid crashes
        self._train_progress_bar = tqdm(disable=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # Retrieve the metrics collected at the end of the epoch
        metrics = trainer.callback_metrics
        postfix_parts = []

        # Collect specific metrics to display in the progress bar
        for key in ["ent_loss", "lit_loss"]:
            value = metrics.get(key)
            if value is not None:
                # Convert tensor values to Python scalars if necessary
                value = value.item() if hasattr(value, "item") else value
                postfix_parts.append(f"{key}: {value:.4f}")

        # Join the metrics into a single string for display
        postfix_str = " | ".join(postfix_parts)
        # Update the progress bar's postfix and increment the epoch count
        self.epoch_bar.set_postfix_str(postfix_str)
        self.epoch_bar.update(1)


class ASWA(Callback):
    """ Adaptive stochastic weight averaging
        ASWE keeps track of the validation performance and update s the ensemble model accordingly.
        """

    def __init__(self, num_epochs, path):
        super().__init__()
        self.path=path
        self.num_epochs=num_epochs
        self.initial_eval_setting = None
        self.epoch_count=0
        self.alphas = []
        self.val_aswa = -1

    def on_fit_end(self, trainer, model):
        # super().on_fit_end(trainer, model)
        if self.initial_eval_setting:
            # ADD this info back
            trainer.evaluator.args.eval_model = self.initial_eval_setting
        
        if trainer.global_rank==trainer.local_rank==0:
            param_ensemble = torch.load(f"{self.path}/aswa.pt", torch.device("cpu"))
            model.kge_model.load_state_dict(param_ensemble)

    @staticmethod
    def compute_mrr(trainer, model) -> float:
        # (2) Enable eval mode.
        model.eval()
        # (3) MRR performance on the validation data of running model.
        device_name = model.device
        model.kge_model.to("cpu")
        last_val_mrr_running_model = trainer.evaluator.eval(dataset=trainer.entity_dataset,
                                                            trained_model=model.kge_model,
                                                            form_of_labelling="EntityPrediction",
                                                            during_training=True)["Val"]["MRR"]
        model.kge_model.to(device_name)
        # (4) Enable train mode.
        model.train()
        return last_val_mrr_running_model

    def get_aswa_state_dict(self, model):
        # (2) Question: Soft update or Rejection?!
        ensemble_state_dict = torch.load(f"{self.path}/aswa.pt", torch.device(model.device))
        # Perform provision parameter update.
        with torch.no_grad():
            for k, parameters in model.state_dict().items():
                if parameters.dtype == torch.float:
                    ensemble_state_dict[k] = (ensemble_state_dict[k] * sum(self.alphas) + parameters) / (1 + sum(self.alphas))
        return ensemble_state_dict

    def decide(self, running_model_state_dict, ensemble_state_dict, val_running_model, mrr_updated_ensemble_model):
        """
        Perform Hard Update, software or rejection

        Parameters
        ----------
        running_model_state_dict
        ensemble_state_dict
        val_running_model
        mrr_updated_ensemble_model

        Returns
        -------

        """
        # (1) HARD UPDATE:
        # If the validation performance of the running model is greater than
        # the validation performance of updated ASWA and
        # the validation performance of ASWA
        if val_running_model > mrr_updated_ensemble_model and val_running_model > self.val_aswa:
            """Hard Update """
            # (1.1) Save the running model as ASWA
            torch.save(running_model_state_dict, f=f"{self.path}/aswa.pt")
            # (2.1) Resect alphas/ensemble weights
            self.alphas.clear()
            # (2.2) Store the validation performance of ASWA
            self.val_aswa = val_running_model
            return True

        # (2) SOFT UPDATE:
        # If the validation performance of the running model is less  than
        # the validation performance of updated ASWA
        if mrr_updated_ensemble_model > self.val_aswa:
            """Soft update"""
            self.val_aswa = mrr_updated_ensemble_model
            torch.save(ensemble_state_dict, f=f"{self.path}/aswa.pt")
            self.alphas.append(1.0)
            return True
        # (3) Rejection:
        if self.val_aswa > mrr_updated_ensemble_model:
            """ Ignore """
            self.alphas.append(0)
            return True

    def on_train_epoch_end(self, trainer, model):
        
        if (trainer.global_rank == trainer.local_rank == 0) is False:
            return None

        # (1) Increment epoch counter
        self.epoch_count += 1
        # (2) Save the given eval setting if it is not saved.
        if self.initial_eval_setting is None:
            self.initial_eval_setting = trainer.evaluator.args.eval_model
            trainer.evaluator.args.eval_model = "val"
        # (3) Compute MRR of the running model.
        val_running_model = self.compute_mrr(trainer, model)

        # (4) Initialize ASWA if it is not initialized.
        if self.val_aswa == -1:
            torch.save(model.kge_model.state_dict(), f=f"{self.path}/aswa.pt")
            self.alphas.append(1.0)
            self.val_aswa = val_running_model
            return True
        else:
            # (5) Load ASWA ensemble parameters.
            ensemble_state_dict = self.get_aswa_state_dict(model.kge_model)
            # (6) Initialize ASWA ensemble with (5).
            ensemble = type(model.kge_model)(model.kge_model.args)
            ensemble.load_state_dict(ensemble_state_dict)
            # (7) Evaluate (6) on the validation data, i.e., perform the lookahead operation.
            mrr_updated_ensemble_model = trainer.evaluator.eval(dataset=trainer.entity_dataset,
                                                                trained_model=ensemble,
                                                                form_of_labelling="EntityPrediction",
                                                                during_training=True)["Val"]["MRR"]
            # print(f"| MRR Running {val_running_model:.4f} | MRR ASWA: {self.val_aswa:.4f} |ASWA|:{sum(self.alphas)}")
            # (8) Decide whether ASWA should be updated via the current running model.
            self.decide(model.kge_model.state_dict(), ensemble_state_dict, val_running_model, mrr_updated_ensemble_model)



class LiteralCallback(Callback):
    """Callback for handling literal embeddings during training."""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        csv_path = args.get('full_storage_path', '')+ '/'+'entity_to_idx.csv'
        if csv_path:
            e2idx_df = pd.read_csv(csv_path)
            entity_to_idx = {row["entity"]: idx for idx, row in e2idx_df.iterrows()}
        self.literal_dataset = LiteralDataset(
            dataset_dir=args.get('dataset_dir', ''),
            ent_idx=entity_to_idx
        )
        self.Literal_model = LiteralEmbeddings(
            num_of_data_properties=self.literal_dataset.num_data_properties,
            embedding_dims=args.get('embedding_dim', '')
        )
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        entiy_ids = batch[0][:, 0].long().to("cpu")
        lit_entities, lit_properties, y_true = self.literal_dataset.get_batch(
            entiy_ids,
            random_seed=None,
        )
        lit_entities, lit_properties, y_true = (
            lit_entities.to(pl_module.device),
            lit_properties.to(pl_module.device),
            y_true.to(pl_module.device),
        )
        # For the old LiteralEmbeddings model, pass indices directly
        yhat_lit = self.Literal_model(lit_entities, lit_properties)
        lit_loss = torch.nn.functional.l1_loss(yhat_lit, y_true)
        pl_module.log("lit_loss", lit_loss, on_step=False, on_epoch=True, prog_bar=True)  

        # Combine with the main model loss (assume it's in outputs["loss"] or similar)
        main_loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else None
        if main_loss is not None:
            combined_loss = main_loss + lit_loss
            pl_module.log("combined_loss", combined_loss, on_step=False, on_epoch=True, prog_bar=True)
            outputs["loss"] = combined_loss  # update the loss for backprop
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    