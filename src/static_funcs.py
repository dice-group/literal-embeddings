import csv
import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from dicee.knowledge_graph_embeddings import KGE
from dicee.static_funcs import intialize_model
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


@dataclass
class KGEModelComponents:
    """
    Data structure to organize and track KGE model components.
    
    Attributes:
        model: The loaded KGE model
        config: Model configuration dictionary
        entity_to_idx: Entity to index mapping
        relation_to_idx: Relation to index mapping
        er_vocab: Entity-relation vocabulary (optional)
        test_set: Test set array (optional)
        model_path: Path to the model directory
    """
    model: Any
    config: Dict
    entity_to_idx: Dict[str, int]
    relation_to_idx: Dict[str, int]
    er_vocab: Optional[Any] = None
    test_set: Optional[np.ndarray] = None
    model_path: Optional[str] = None
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension from the model."""
        return self.model.embedding_dim
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.model.name
    
    @property
    def num_entities(self) -> int:
        """Get the number of entities."""
        return len(self.entity_to_idx)
    
    @property
    def num_relations(self) -> int:
        """Get the number of relations."""
        return len(self.relation_to_idx)
    
    def get_entity_idx(self, entity: str) -> int:
        """Get index for an entity."""
        return self.entity_to_idx.get(entity, -1)
    
    def get_relation_idx(self, relation: str) -> int:
        """Get index for a relation."""
        return self.relation_to_idx.get(relation, -1)
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the model components."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "model_path": self.model_path,
            "has_er_vocab": self.er_vocab is not None,
            "has_test_set": self.test_set is not None
        }


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
    device = literal_model.device if hasattr(literal_model, 'device') else "cpu"
    entities, properties = entities.to(device), properties.to(device)

    with torch.no_grad():
        # Check if this is an external embedding model (doesn't have entity_embeddings parameter)
        if hasattr(literal_model, 'entity_embeddings'):
            # Old model - pass entities directly
            predictions = literal_model.forward(entities, properties)
        else:
            # External embedding model - get embeddings from KGE model first
            entity_embeddings = model.entity_embeddings(entities)
            # Ensure embeddings are on the same device as the literal model
            entity_embeddings = entity_embeddings.to(device)
            predictions = literal_model.forward(entity_embeddings, properties)

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
    

def load_model_components(kge_path: str) -> Optional[KGEModelComponents]:
    """Load configuration and weights for a KGE model or a direct KGE model.

    Args:
        kge_path (str): The path to the directory containing the model's files.

    Returns:
        Optional[KGEModelComponents]: A KGEModelComponents object containing all model 
        components, or None if loading fails.

    Raises:
        FileNotFoundError: If one or more required files do not exist.
        Exception: For any other error that occurs during the loading process.
    """
    kge_model = None
    config = None
    entity_to_idx = None
    relation_to_idx = None
    er_vocab = None
    test_set = None
    
    try:
        kge_obj = KGE(path=kge_path)
        kge_model = kge_obj.model
        config = kge_obj.configs
        entity_to_idx = kge_obj.entity_to_idx
        relation_to_idx = kge_obj.relation_to_idx
        print("DiCE KGE model loaded successfully!")
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
            print("Manual KGE load successful!")

        except Exception as e:
            print(
                "Building the KGE model failed, check pre-trained KGE directory", str(e)
            )
            return None
    
    # Load optional components
    try:
        er_vocab_path = os.path.join(kge_path, "er_vocab.p")
        if os.path.exists(er_vocab_path):
            er_vocab = pickle.load(open(er_vocab_path, "rb"))
    except Exception as e:
        print(f"Warning: Could not load er_vocab: {e}")
    
    try:
        test_set_path = os.path.join(kge_path, "test_set.npy")
        if os.path.exists(test_set_path):
            test_set = np.load(test_set_path, allow_pickle=True)
    except Exception as e:
        print(f"Warning: Could not load test_set: {e}")
    
    # Create and return the KGEModelComponents object
    model_components = KGEModelComponents(
        model=kge_model,
        config=config,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        er_vocab=er_vocab,
        test_set=test_set,
        model_path=kge_path
    )
    return model_components


@torch.no_grad()
def evaluate_link_prediction_performance_with_reciprocals(model, triples,
                                                          model_components =None):
    model.eval()
    entity_to_idx = model_components.entity_to_idx
    relation_to_idx = model_components.relation_to_idx
    batch_size = model_components.config["batch_size"]
    num_triples = len(triples)
    er_vocab = model_components.er_vocab
    ranks = []
    # Hit range
    hits_range = [i for i in range(1, 11)]
    hits = {i: [] for i in hits_range}
    # Iterate over integer indexed triples in mini batch fashion
    for i in range(0, num_triples, batch_size):
        # (1) Get a batch of data.
        str_data_batch = triples[i:i + batch_size]
        data_batch = np.array(
            [[entity_to_idx[str_triple[0]], relation_to_idx[str_triple[1]], entity_to_idx[str_triple[2]]] for
             str_triple in str_data_batch])
        # (2) Extract entities and relations.
        e1_idx_r_idx, e2_idx = torch.LongTensor(data_batch[:, [0, 1]]), torch.tensor(data_batch[:, 2])
        # (3) Predict missing entities, i.e., assign probs to all entities.
        predictions = model(e1_idx_r_idx)
        # (4) Filter entities except the target entity
        for j in range(data_batch.shape[0]):
            # (4.1) Get the ids of the head entity, the relation and the target tail entity in the j.th triple.
            str_h, str_r, str_t = str_data_batch[j]

            id_e, id_r, id_e_target = data_batch[j]
            # (4.2) Get all ids of all entities occurring with the head entity and relation extracted in 4.1.
            filt = [entity_to_idx[_] for _ in er_vocab[(str_h, str_r)]]
            # (4.3) Store the assigned score of the target tail entity extracted in 4.1.
            target_value = predictions[j, id_e_target].item()
            # (4.4.1) Filter all assigned scores for entities.
            predictions[j, filt] = -np.Inf
            # (4.5) Insert 4.3. after filtering.
            predictions[j, id_e_target] = target_value
        # (5) Sort predictions.
        sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
        # (6) Compute the filtered ranks.
        for j in range(data_batch.shape[0]):
            # index between 0 and \inf
            rank = torch.where(sort_idxs[j] == e2_idx[j])[0].item() + 1
            ranks.append(rank)
            for hits_level in hits_range:
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
    # (7) Sanity checking: a rank for a triple
    assert len(triples) == len(ranks) == num_triples
    hit_1 = sum(hits[1]) / num_triples
    hit_3 = sum(hits[3]) / num_triples
    hit_10 = sum(hits[10]) / num_triples
    mean_reciprocal_rank = np.mean(1. / np.array(ranks))

    results = {'H@1': hit_1, 'H@3': hit_3, 'H@10': hit_10, 'MRR': mean_reciprocal_rank}
    return results


def get_full_storage_path(args):
    """
    Generate a full storage path for experiments based on args configuration.
    
    Args:
        args: Arguments namespace containing experiment configuration
        
    Returns:
        str: Full path for storing experiment results
    """
    if args.full_storage_path:
        # If explicitly set, use as-is
        return args.full_storage_path
    
    # Extract dataset name from dataset_dir or path_single_kg
    dataset_name = None
    if hasattr(args, 'dataset_dir') and args.dataset_dir:
        dataset_name = os.path.basename(args.dataset_dir.rstrip('/'))
    elif hasattr(args, 'path_single_kg') and args.path_single_kg:
        dataset_name = os.path.basename(args.path_single_kg.rstrip('/'))
    
    if not dataset_name:
        dataset_name = "unknown_dataset"
    
    # Generate timestamp for test runs
    if getattr(args, 'test_runs', False):
        exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        return f"Test_runs/{exp_time}"
    
    # Determine experiment type and create appropriate path structure
    if getattr(args, 'combined_training', False):
        # Combined KGE + Literal training
        base_path = f"Experiments/KGE_Combined/{dataset_name}_combined"
        model_name = getattr(args, 'model', 'unknown_model')
        return f"{base_path}/{model_name}"
    
    elif getattr(args, 'literal_training', False):
        # Literal-only training
        base_path = f"Experiments/Literal/{dataset_name}"
        
        # Add model-specific information
        literal_model = getattr(args, 'literal_model', 'mlp')
        embedding_dim = getattr(args, 'embedding_dim', 'unknown_dim')
        lit_norm = getattr(args, 'lit_norm', 'z-norm')
        
        # Include special configurations in path
        config_parts = [literal_model, str(embedding_dim), lit_norm]
        
        if getattr(args, 'gate_residual', False):
            config_parts.append('gated')
        if getattr(args, 'residual_connection', False):
            config_parts.append('residual')
        if getattr(args, 'freeze_entity_embeddings', False):
            config_parts.append('frozen')
            
        config_str = '_'.join(config_parts)
        return f"{base_path}/{config_str}"
    
    else:
        # Standard KGE training
        base_path = f"Experiments/KGE/{dataset_name}"
        model_name = getattr(args, 'model', 'unknown_model')
        embedding_dim = getattr(args, 'embedding_dim', 'unknown_dim')
        return f"{base_path}/{model_name}_{embedding_dim}"
    

def create_training_report(args, model, dataset, start_time, end_time, additional_info=None):
    """
    Create a training report with the exact same structure as the existing format.
    
    Args:
        args: Training arguments
        model: Trained model
        dataset: Dataset object
        start_time: Training start time
        end_time: Training end time
        additional_info: Additional information to include in report
    """
    
    # Calculate model size and parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes per param)
    
    # Extract dataset information
    num_entities = getattr(dataset, 'num_entities', None)
    num_relations = getattr(dataset, 'num_relations', None)
    num_train_triples = len(dataset.train_set) if hasattr(dataset, 'train_set') else None
    
    # Calculate runtime
    runtime_seconds = end_time - start_time
    
    # Build the report with exact same structure
    report = {
        "num_train_triples": num_train_triples,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "max_length_subword_tokens": None,
        "runtime_kg_loading": getattr(args, '_kg_loading_time', None),
        "EstimatedSizeMB": model_size_mb,
        "NumParam": total_params,
        "path_experiment_folder": getattr(args, 'full_storage_path', None),
        "Runtime": runtime_seconds
    }
    
    return report

def save_training_report(report, save_path):
    """Save the training report to a JSON file"""
    report_path = os.path.join(save_path, "report.json")
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Training report saved to: {report_path}")
    return report_path