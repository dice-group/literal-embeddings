import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_lit_preds(
    literal_dataset, dataset_type: str, model, literal_model, device
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

    if dataset_type == "test":
        dataset_path = literal_dataset.test_file_path
    elif dataset_type == "val":
        dataset_path = literal_dataset.val_file_path
    else:
        raise ValueError("Invalid dataset_type. Choose from 'train', 'test', or 'val'.")

    dataset_df = pd.read_csv(
        dataset_path, sep="\t", header=None, names=["head", "relation", "tail"]
    )

    dataset_df["head_idx"] = dataset_df["head"].map(literal_dataset.ent_idx)
    dataset_df["rel_idx"] = dataset_df["relation"].map(
        literal_dataset.data_property_to_idx
    )

    entities = torch.LongTensor(dataset_df["head_idx"].values).to(device)
    properties = torch.LongTensor(dataset_df["rel_idx"].values).to(device)

    model.eval()
    literal_model.eval()

    with torch.no_grad():
        entity_embeddings = model.entity_embeddings(entities)
        predictions = literal_model.forward(entity_embeddings, properties)

    dataset_df["preds"] = predictions.cpu().numpy()

    def denormalize(
        row,
        normalization_params,
    ):
        type_stats = normalization_params[row["relation"]]
        return (row["preds"] * type_stats["std"]) + type_stats["mean"]

    dataset_df["denormalized_preds"] = dataset_df.apply(
        denormalize, axis=1, args=(literal_dataset.normalization_params,)
    )

    # Compute MAE and RMSE for each relation
    def compute_errors(group):
        actuals = group["tail"]
        predictions = group["denormalized_preds"]
        mae = mean_absolute_error(actuals, predictions)
        rmse = root_mean_squared_error(actuals, predictions)
        return pd.Series({"MAE": mae, "RMSE": rmse})

    error_metrics = dataset_df.groupby("relation").apply(compute_errors).reset_index()
    pd.options.display.float_format = "{:.6f}".format  # 6 decimal places
    print("Literal Prediction Results on Test Set")
    print(error_metrics)
    return error_metrics


def combine_losses(loss1, loss2, method="min-max", lambda1=0.2, lambda2=0.8, eps=1e-8):
    """
    Combines two loss values using normalization.

    Args:
        loss1 (torch.Tensor): First loss value.
        loss2 (torch.Tensor): Second loss value.
        method (str): Normalization method ('min-max' or 'mean').
        lambda1 (float): Weight for first loss.
        lambda2 (float): Weight for second loss.
        eps (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: Combined normalized loss.
    """

    if method == "min-max":
        # Normalize using Min-Max scaling
        min_loss = min(loss1.item(), loss2.item())
        max_loss = max(loss1.item(), loss2.item())
        norm_loss1 = (loss1 - min_loss) / (max_loss - min_loss + eps)
        norm_loss2 = (loss2 - min_loss) / (max_loss - min_loss + eps)

    elif method == "mean":
        # Normalize using mean normalization
        mean_loss = (loss1 + loss2) / 2
        norm_loss1 = loss1 / (mean_loss + eps)
        norm_loss2 = loss2 / (mean_loss + eps)

    combined_loss = lambda1 * norm_loss1 + lambda2 * norm_loss2
    return combined_loss
