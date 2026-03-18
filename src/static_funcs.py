import csv
import json
import os
from datetime import datetime

import pandas as pd
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import (
    StochasticWeightAveraging as SWA,
)
from torch.utils.data import DataLoader

from dicee.static_funcs import intialize_model
from src.callbacks import ASWA, EpochLevelProgressBar, PeriodicEvalCallback
from src.clifford import CLNN_KGE
from src.clifford_conv import CliffConvKGE
from src.dataset import KvsAll, OnevsAllDataset


def extract_metrics(data, split_name):
        metrics = data.get(split_name, {})
        return [
            metrics.get("MRR", ""), metrics.get("H@1", ""), 
            metrics.get("H@3", ""), metrics.get("H@10", "")
        ]

def save_kge_experiments(args, loss_log=None):

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


def get_callbacks(args):
    callbacks = [EpochLevelProgressBar()]
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="ent_loss_val",
                patience=args.early_stopping_patience,
                mode="min",
            )
        )
    if args.swa:
        callbacks.append(SWA(swa_epoch_start=1, swa_lrs=0.05))
    elif args.adaptive_swa:
        callbacks.append(ASWA(num_epochs=args.num_epochs, path=args.full_storage_path))
    else:
        print("No Stochastic Weight Averaging (SWA) or Adaptive SWA (ASWA) used.")

    if args.eval_every_n_epochs > 0 or args.eval_at_epochs is not None:
        callbacks.append(
            PeriodicEvalCallback(
                experiment_path=args.full_storage_path,
                max_epochs=args.num_epochs,
                eval_every_n_epoch=args.eval_every_n_epochs,
                eval_at_epochs=args.eval_at_epochs,
                save_model_every_n_epoch=args.save_every_n_epochs,
                n_epochs_eval_model=args.n_epochs_eval_model,
            )
        )
    return callbacks


def collate_fn(batch):
    xs, y_vecs, targets = zip(*batch)
    xs = torch.stack(xs)
    y_vecs = torch.stack(y_vecs)
    if isinstance(targets, tuple) and isinstance(targets[0], list):
        targets = [torch.as_tensor(t, dtype=torch.long) for t in targets]
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.stack(targets)
    return xs, y_vecs, targets


def get_dataloaders(args, entity_dataset):
    train_dataset = None
    valid_dataset = None

    if args.scoring_technique == "KvsAll":
        train_dataset = KvsAll(
            train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx,
        )
        if args.log_validation and not args.train_all_triples:
            valid_dataset = KvsAll(
                train_set_idx=entity_dataset.valid_set,
                entity_idxs=entity_dataset.entity_to_idx,
            )
    elif args.scoring_technique == "1vsAll":
        train_dataset = OnevsAllDataset(
            train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx,
        )
        if args.log_validation and not args.train_all_triples:
            valid_dataset = OnevsAllDataset(
                train_set_idx=entity_dataset.valid_set,
                entity_idxs=entity_dataset.entity_to_idx,
            )
    else:
        raise ValueError(
            f"Unknown scoring technique: {args.scoring_technique}. "
            "Supported techniques: 'KvsAll', 'OneVsAll'"
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_core,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    valid_dataloader = None
    if valid_dataset is not None:
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=7,
        )

    return train_dataloader, valid_dataloader


def get_model(args, entity_dataset=None):
    if args.model == "CLNN_KGE":
        return CLNN_KGE(args=vars(args), entity2idx=entity_dataset.entity_to_idx)
    if args.model == "CliffConvKGE":
        return CliffConvKGE(args=vars(args), entity2idx=entity_dataset.entity_to_idx)
    kge_model, _ = intialize_model(vars(args), 0)
    return kge_model

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
        return f"Experiments/Test_runs/{exp_time}"

    base_path = f"Experiments/KGE/{dataset_name}"
    model_name = getattr(args, 'model', 'unknown_model')
    embedding_dim = getattr(args, 'embedding_dim', 'unknown_dim')
    return f"{base_path}/{model_name}_{embedding_dim}"
    

def create_and_save_report(trainer):
    num_entities = trainer.kge_model.num_entities
    num_relations = trainer.kge_model.num_relations
    train_triples = len(trainer.trainer.entity_dataset.train_set)
    # Calculate model size and parameters
    total_params = sum(p.numel() for p in trainer.kge_model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes per param)
    path_experiment = trainer.args.full_storage_path
    run_time = datetime.now() - trainer.start_time
    runtime_seconds = run_time.total_seconds()
    print("Total runtime of the experiment", runtime_seconds)
    # Create the report
    report = {
        "num_train_triples": train_triples,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "max_length_subword_tokens": None,
        "runtime_kg_loading": getattr(trainer.args, '_kg_loading_time', None),
        "EstimatedSizeMB": model_size_mb,
        "NumParam": total_params,
        "path_experiment_folder": path_experiment,
        "Runtime": runtime_seconds
    }
    report_path = f"{path_experiment}/report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
