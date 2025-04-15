import json
import os
import pickle
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from dicee.knowledge_graph_embeddings import KGE
from dicee.static_funcs import intialize_model
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import torch.nn as nn


def denormalize(row, normalization_params, norm_type="z-norm"):
    if norm_type is None:
        return row["preds_norm"]
    type_stats = normalization_params[row["rel_idx"]]
    if norm_type == "z-norm":
        return (row["preds_norm"] * type_stats["std"]) + type_stats["mean"]

    elif norm_type == "min-max":
        return (
            row["preds_norm"] * (type_stats["max"] - type_stats["min"])
        ) + type_stats["min"]

    else:
        raise ValueError(
            "Unsupported normalization type. Use 'z-norm','min-max' or None."
        )


# Compute MAE and RMSE for each relation
def compute_errors(group):
    actuals = group["tail"]
    predictions = group["preds"]
    mae = mean_absolute_error(actuals, predictions)
    rmse = root_mean_squared_error(actuals, predictions)
    return pd.Series({"MAE": mae, "RMSE": rmse})


def evaluate_lit_preds(
    literal_dataset,
    dataset_type: str,
    model,
    literal_model,
    device: None,
    multi_regression=False,
):
    """
    Evaluates the model on the specified dataset.

    Parameters:
    - dataset_type: Type of dataset to evaluate ('train', 'test', or 'val')
    - model: Trained main model
    - literal_model: Trained literal model
    - device: Device to perform computation (e.g., 'cpu' or 'cuda')

    Returns:
    - DataFrame with MAE and RMSE metrics for each relation
    """

    target_df = literal_dataset.get_df(split=dataset_type)

    entities = torch.LongTensor(target_df["head_idx"].values)
    properties = torch.LongTensor(target_df["rel_idx"].values)

    model.eval()
    literal_model.eval()

    with torch.no_grad():
        entity_embeddings = model.entity_embeddings(entities)
        entity_embeddings, properties = entity_embeddings.to(device), properties.to(
            device
        )
        predictions = literal_model.forward(entity_embeddings, properties)

    if multi_regression:
        target_df["preds_norm"] = (
            predictions.gather(1, properties.view(-1, 1)).cpu().numpy()
        )
    else:
        target_df["preds_norm"] = predictions.cpu().numpy()

    target_df["preds"] = target_df.apply(
        denormalize,
        axis=1,
        args=(
            literal_dataset.normalization_params,
            literal_dataset.normalization,
        ),
    )

    error_metrics = target_df.groupby("relation").apply(compute_errors).reset_index()
    pd.options.display.float_format = "{:.6f}".format  # 6 decimal places
    print("Literal Prediction Results on Test Set")
    print(error_metrics)
    return error_metrics


def load_model_components(kge_path: str) -> Tuple[Any, Dict]:
    """Load configuration and weights for a KGE model or a direct KGE model.

    Args:
        path (str): The path to the directory containing the model's files.

    Returns:
        Tuple[Any, Dict]: A tuple containing the loaded model and its configuration.

    Raises:
        FileNotFoundError: If one or more required files do not exist.
        Exception: For any other error that occurs during the loading process.
    """
    try:
        kge_obj = KGE(path=kge_path)
        kge_model = kge_obj.model
        config = kge_obj.configs
        entity_to_idx = kge_obj.entity_to_idx
        relation_to_idx = kge_obj.relation_to_idx
    except Exception as e:
        print("Cannot load as dicee KGE model.", str(e), "Trying manual load.")
        try:
            config_path = os.path.join(kge_path, "configuration.json")
            model_path = os.path.join(kge_path, "model.pt")

            if os.path.isfile(kge_path + "/entity_to_idx.p"):
                entity_to_idx_path = os.path.join(kge_path, "entity_to_idx.p")
                with open(entity_to_idx_path, "rb") as f:
                    entity_to_idx = pickle.load(f)
            elif os.path.isfile(kge_path + "/entity_to_idx.csv"):
                entity_to_idx_path = os.path.join(kge_path, "entity_to_idx.csv")
                e2idx_df = pd.read_csv(entity_to_idx_path)
                entity_to_idx = {row["entity"]: idx for idx, row in e2idx_df.iterrows()}
            else:
                entity_to_idx_path = None

            if os.path.isfile(kge_path + "/relation_to_idx.p"):
                relation_to_idx_path = os.path.join(kge_path, "relation_to_idx.p")
                with open(relation_to_idx_path, "rb") as f:
                    relation_to_idx = pickle.load(f)
            elif os.path.isfile(kge_path + "/relation_to_idx.csv"):
                relation_to_idx_path = os.path.join(kge_path, "relation_to_idx.csv")
                r2idx_df = pd.read_csv(relation_to_idx_path)
                relation_to_idx = {
                    row["relation"]: idx for idx, row in r2idx_df.iterrows()
                }
            else:
                relation_to_idx_path = None

            if not all(os.path.exists(file) for file in [config_path, model_path]):
                raise FileNotFoundError("One or more required files do not exist.")

            with open(config_path, "r") as f:
                config = json.load(f)

            weights = torch.load(model_path, map_location="cpu")

            kge_model, _ = intialize_model(config, 0)
            kge_model.load_state_dict(weights)

        except Exception as e:
            print(
                "Building the KGE model failed, check pre-trained KGE directory", str(e)
            )
            exit(0)
        print("Manual KGE load Successfull!!")
    return kge_model, config, entity_to_idx, relation_to_idx


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable log variances (log σ²) for numerical stability and positivity
        self.log_sigma_ent = nn.Parameter(torch.tensor(0.0))  # for entity loss
        self.log_sigma_lit = nn.Parameter(torch.tensor(0.0))  # for literal loss

    def forward(self, loss_ent, loss_lit):
        # Uncertainty-weighted total loss
        total_loss = (
            torch.exp(-self.log_sigma_ent) * loss_ent
            + torch.exp(-self.log_sigma_lit) * loss_lit
            + self.log_sigma_ent
            + self.log_sigma_lit
        )
        return total_loss
