import gc
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import LiteralDataset
from src.model import LiteralEmbeddings, LiteralEmbeddingsClifford


def clear_cuda_cache():
    """Clear CUDA cache and force garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
def reset_random_seeds(seed, run=0):
    """Reset all random seeds for reproducibility."""
    torch.manual_seed(seed + run)
    np.random.seed(seed + run)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + run)
        # Reset CUDA deterministic settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_best_configs(config_path=None):
    """Load the best configurations from the config file if it exists."""
    if config_path is None:
        config_path = "Stats/best_configs.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            return None
    else:
        print(f"Config file {config_path} not found. Using default parameters.")
        return None

def apply_best_config_to_args(args):
    """Apply the best configuration for the dataset if available."""
    config_path = args.config_path if hasattr(args, 'config_path') else None
    configs = load_best_configs(config_path)
    if configs is None:
        return args
    
    # Extract dataset name from dataset_dir
    dataset_name = os.path.basename(args.dataset_dir)
    
    if dataset_name in configs:
        config = configs[dataset_name]
        print(f"Applying best configuration for dataset: {dataset_name}")
        
        # Apply configuration parameters
        if hasattr(args, 'batch_size') and 'batch_size' in config:
            args.batch_size = config['batch_size']
            print(f"  Set batch_size: {args.batch_size}")
            
        if hasattr(args, 'lit_epochs') and 'lit_epochs' in config:
            args.lit_epochs = config['lit_epochs']
            print(f"  Set lit_epochs: {args.lit_epochs}")
            
        if hasattr(args, 'lit_lr') and 'lit_lr' in config:
            args.lit_lr = config['lit_lr']
            print(f"  Set lit_lr: {args.lit_lr}")
    else:
        print(f"No configuration found for dataset: {dataset_name}. Using default parameters.")
    
    return args

def get_full_storage_path_literals(args, dataset_name):
    """Construct the full storage path for literal experiments."""
    if not args.full_storage_path:
        # Generate timestamp for test runs
        if getattr(args, 'test_runs', False):
            exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            return f"Experiments/Test_runs/Literals/{exp_time}"
        
        else:
            full_path = (
            f"Experiments/Literals/{dataset_name}/{args.model}_{args.embedding_dim}_{args.literal_model}"
            )
            if  args.update_entity_embeddings:
                full_path += "_emb_updated"
            if args.no_residual:
                full_path += "_no_res"

    
    return full_path

    
def get_literal_datasets(args, entity_to_idx):
    """ Create and return literal dataset and dataloader """
    literal_dataset = LiteralDataset(dataset_dir=args.dataset_dir, ent_idx=entity_to_idx,
                                      normalization=args.lit_norm, sampling_ratio=args.lit_sampling_ratio)

    literal_dataloader = DataLoader(literal_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_core, pin_memory=args.device.type == "cuda")

    return literal_dataset, literal_dataloader


def get_litem_model(args, literal_dataset, kge_model, run ):
    # Initialize the appropriate literal model based on args.literal_model
    # freeze entity embeddings if the flag is set
    clear_cuda_cache()
    reset_random_seeds(args.random_seed, run)

    freeze_entity_embeddings = not args.update_entity_embeddings
    if args.literal_model == 'clifford':
        literal_model = LiteralEmbeddingsClifford(
            num_of_data_properties=literal_dataset.num_data_properties,
            embedding_dims=kge_model.embedding_dim,
            entity_embeddings=kge_model.entity_embeddings,
            freeze_entity_embeddings=freeze_entity_embeddings,
            gate_residual=args.gate_residual,
            dropout=getattr(args, 'dropout', 0.15),
        )
    elif args.literal_model == 'mlp':
        literal_model = LiteralEmbeddings(
            num_of_data_properties=literal_dataset.num_data_properties,
            embedding_dims=kge_model.embedding_dim,
            entity_embeddings=kge_model.entity_embeddings,
            freeze_entity_embeddings=freeze_entity_embeddings,
            dropout=getattr(args, 'dropout', 0.3),
            gate_residual=getattr(args, 'gate_residual', True),
        )
    else:
        raise ValueError(f"Unknown literal model type: {args.literal_model}. Supported types: 'mlp', 'clifford'")
    return literal_model


def get_final_results_df(lit_results: list, lit_loss: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert literal results and losses from multiple runs into DataFrames.

    - If only 1 run is given, column names remain as-is.
    - If multiple runs are given, columns get suffixed with `_run_X`.
    """
    final_results_df = None
    final_loss_df = None

    n_runs = len(lit_results)  # number of runs

    for run in range(n_runs):
        # --- Results ---
        df_results = pd.DataFrame(lit_results[run])
        if n_runs > 1:  # add suffix only if multiple runs
            df_results.rename(
                columns={col: f"{col}_run_{run+1}" for col in df_results.columns if col != "relation"},
                inplace=True,
            )

        if final_results_df is None:
            final_results_df = df_results.copy()
        else:
            final_results_df = pd.merge(final_results_df, df_results, on="relation", how="left")

        # --- Loss ---
        df_loss = pd.DataFrame(lit_loss[run])
        if n_runs > 1:
            df_loss.rename(columns={"lit_loss": f"lit_loss_run_{run+1}"}, inplace=True)

        if final_loss_df is None:
            final_loss_df = df_loss.copy()
            final_loss_df.insert(0, "epoch", range(1, len(df_loss) + 1))
        else:
            final_loss_df = pd.concat([final_loss_df, df_loss], axis=1)

    return final_results_df, final_loss_df


def save_literal_experiments(args=None, literal_model=None, lit_results=None, lit_losses=None, attr_to_idx=None):
    if args is None:
        raise ValueError("args must be provided to save experiment artifacts.")

    # Ensure output directory exists
    os.makedirs(args.full_storage_path, exist_ok=True)
    results_df, loss_df = get_final_results_df(lit_results, lit_losses)

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

    if attr_to_idx is not None:
        idx_to_attr = {v: k for k, v in attr_to_idx.items()}
        df = pd.DataFrame.from_dict(idx_to_attr, orient="index", columns=["attribute"])
        df.to_csv(args.full_storage_path + "/attribute_to_idx.csv")
        print(f"Literal attributes indexing saved to {args.full_storage_path}/attribute_to_idx.csv")

    # Save training loss log (if available)
    if loss_df is not None:
        if isinstance(loss_df, dict):
            loss_df = pd.DataFrame.from_dict(loss_df, orient="columns")

        loss_path = os.path.join(args.full_storage_path, "lit_loss_log.csv")
        loss_df.to_csv(loss_path, index=False)

    print("Literal Experiments saved at", args.full_storage_path)
    

# if not args.freeze_entity_embeddings:
    #     test_dir = os.path.join(args.dataset_dir, "test.txt")
    #     assert os.path.exists(test_dir), f"Test file not found at {test_dir}"
    #     test_triples = pd.read_csv(test_dir,
    #                        sep="\s+",
    #                        header=None, usecols=[0, 1, 2],
    #                        names=['subject', 'relation', 'object'],
    #                        dtype=str).values.tolist()
    #     lp_results_kge = evaluate_link_prediction_performance_with_reciprocals(kge_model, triples=test_triples,
    #                                                                         model_components=model_components)
    #     print(f"Link prediction results before literal training:\n {lp_results_kge}")
    #     # Save the model with updated entity embeddings
    #     kge_model.entity_embeddings.weight.data.copy_(literal_model.entity_embeddings.weight.data)

    #     save_checkpoint_model(
    #         model=kge_model,
    #         path =args.full_storage_path + f"/lit_augmented_model.pt",
    #     )
        
    #     lp_results_kge_lit = evaluate_link_prediction_performance_with_reciprocals(kge_model, triples=test_triples,
    #                                                                         model_components=model_components)
    #     print(f"Link prediction results after literal training:\n {lp_results_kge_lit}")