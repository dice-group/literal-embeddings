import csv
import json
import os
import pickle
from typing import Any, Dict, Tuple

import pandas as pd
import torch
from dicee.knowledge_graph_embeddings import KGE
from dicee.static_funcs import intialize_model
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def extract_metrics(data, split_name):
        metrics = data.get(split_name, {})
        return [
            metrics.get("MRR", ""), metrics.get("H@1", ""), 
            metrics.get("H@3", ""), metrics.get("H@10", "")
        ]

def save_literal_experiments(args=None, literal_model=None, results_df=None, loss_df=None):
    if args is None:
        raise ValueError("args must be provided to save experiment artifacts.")

    # Ensure output directory exists
    os.makedirs(args.full_storage_path, exist_ok=True)

    # Save experiment configuration
    # Ensure device is JSON serializable
    if hasattr(args, 'device'):
        args.device = str(args.device)
    exp_configs = vars(args)
    config_path = os.path.join(args.full_storage_path, "configuration.json")
    with open(config_path, "w") as f:
        json.dump(exp_configs, f, indent=4)

    # Save aggregated literal prediction results (if available)
    if results_df is not None:
        results_path = os.path.join(args.full_storage_path, "lit_eval_results.csv")
        results_df.to_csv(results_path, index=False)
    
    # Save literal model state (if provided)
    if literal_model is not None:
        model_path = os.path.join(args.full_storage_path, "literal_model.pt")
        torch.save(literal_model.state_dict(), model_path)

    # Save training loss log (if available)
    if loss_df is not None:
        if isinstance(loss_df, dict):
            loss_df = pd.DataFrame.from_dict(loss_df, orient="columns")

        loss_path = os.path.join(args.full_storage_path, "lit_loss_log.csv")
        loss_df.to_csv(loss_path, index=False)

    print("Literal Experiments saved at", args.full_storage_path)


def save_kge_experiments(args, loss_log=None, lit_results=None):

    if args is None:
        raise ValueError("`args` must be provided to save experiment results.")

    # Create storage directory
    os.makedirs(args.full_storage_path, exist_ok=True)

    # Save loss log (TSV format)
    if loss_log is not None:
        df_loss_log = pd.DataFrame.from_dict(loss_log, orient="index").transpose()
        loss_log_path = os.path.join(args.full_storage_path, "loss_log.tsv")
        df_loss_log.to_csv(loss_log_path, sep="\t", index=False)

    # Save experiment configuration
    args.device = str(args.device)  # Ensure JSON serializable
    config_path = os.path.join(args.full_storage_path, "configuration.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # Save literal prediction results (if provided)
    if lit_results is not None:
        results_path = os.path.join(args.full_storage_path, "lit_results.json")
        lit_results.to_json(results_path, orient="records", indent=4)

    # print(f"\n Experiment results saved at: {args.full_storage_path}")

def log_exp(file_path : str= None, args = None):
    if file_path is None:
        file_path = "Experiments/exp_log.csv"
    if args is None:
        print("No args to log. Abort")
        return
    report_path = f'{args.full_storage_path}/eval_report.json'
                 # Load combined evaluation data
    with open(report_path, 'r') as file:
        report = json.load(file)
    dataset = str(args.dataset_dir).split('/')[-1]
    row_dict = {
        "Model": args.model,
        "Dataset": dataset,
        "lr": args.lr,
        "Embedding_dim": args.embedding_dim,
        "Epochs" : args.num_epochs,
        "p": args.p,
        "q": args.q,
        "r": args.r,
    }
    metric_names = ["MRR", "H@1", "H@3", "H@10"]
    splits = ["Train", "Test", "Val"]

    for split in splits:
        metrics = extract_metrics(report, split)  # Assuming this returns a list in [MRR, H@1, H@3, H@10]
        for metric_name, value in zip(metric_names, metrics):
            row_dict[f"{split}_{metric_name}"] = value
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    # Append the new row
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row_dict.keys())
        writer.writerow(row_dict)
    
    print("Experiments Logged to the file")


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
    device = literal_model.device if hasattr(literal_model, 'device') else device
    entities, properties = entities.to(device), properties.to(device)

    with torch.no_grad():
        predictions = literal_model.forward(entities, properties)

    if multi_regression:
        preds_norm = (
            predictions.gather(1, properties.view(-1, 1)).cpu().numpy()
        )
    else:
        preds_norm = predictions.cpu().numpy()

    attributes = target_df["relation"].to_list()
    target_df["predictions"] = literal_dataset.denormalize(preds_norm=preds_norm,attributes= attributes,
                                                      normalization_params=literal_dataset.normalization_params)

    attr_error_metrics = target_df.groupby("relation").agg(
    MAE=("tail", lambda x: mean_absolute_error(x, target_df.loc[x.index, "predictions"])),
    RMSE=("tail", lambda x: root_mean_squared_error(x, target_df.loc[x.index, "predictions"]))
    ).reset_index()

    pd.options.display.float_format = "{:.6f}".format
    print("Literal-Prediction evaluation results  on Test Set")
    print(attr_error_metrics)
    return attr_error_metrics
    

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
    except Exception:
        # print("Cannot load as dicee KGE model.", str(e), "Trying manual load.")
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
            return None
        print("Manual KGE load Successfull!!")
    return kge_model, config, entity_to_idx, relation_to_idx