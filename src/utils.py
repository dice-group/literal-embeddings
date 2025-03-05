import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def denormalize(row, normalization_params, norm_type="z-norm"):
    type_stats = normalization_params[row["rel_idx"]]

    if norm_type == "z-norm":
        return (row["preds"] * type_stats["std"]) + type_stats["mean"]

    elif norm_type == "min-max":
        return (row["preds"] * (type_stats["max"] - type_stats["min"])) + type_stats[
            "min"
        ]

    else:
        raise ValueError("Unsupported normalization type. Use 'z-norm' or 'min-max'.")


# Compute MAE and RMSE for each relation
def compute_errors(group):
    actuals = group["tail"]
    predictions = group["denormalized_preds"]
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

    entities = torch.LongTensor(target_df["head_idx"].values).to(device)
    properties = torch.LongTensor(target_df["rel_idx"].values).to(device)

    model.eval()
    literal_model.eval()

    with torch.no_grad():
        entity_embeddings = model.entity_embeddings(entities)
        predictions = literal_model.forward(entity_embeddings, properties)

    if multi_regression:
        target_df["preds"] = predictions.gather(1, properties.view(-1, 1)).cpu().numpy()
    else:
        target_df["preds"] = predictions.cpu().numpy()

    target_df["denormalized_preds"] = target_df.apply(
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
