### Main File
import argparse
import json
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from dicee.config import Namespace
from dicee.dataset_classes import KvsAll
from dicee.evaluator import Evaluator
from dicee.knowledge_graph import KG
from dicee.knowledge_graph_embeddings import KGE
from dicee.models import Keci, TransE
from dicee.static_funcs import read_or_load_kg, store
from torch.utils.data import DataLoader

from src.trainer import train_literal_model, train_model
from src.utils import evaluate_lit_preds

# Configuration setup
args = Namespace()
args.scoring_technique = "KvsAll"
args.dataset_dir = "KGs/FamilyT"
args.eval_model = "train_test_eval"
args.apply_reciprical_or_noise = True
args.neg_ratio = 0
args.label_smoothing_rate = 0.0
args.batch_size = 1024
args.normalization = None
args.num_epochs = 150
args.embedding_dim = 128
args.lr = 0.05
args.lit_dataset_dir = "KGs/FamilyL"
args.optimize_with_literals = True
args.lit_lr = 0.001
args.save_embeddings_as_csv = False
args.save_experiment = False


def update_args_with_cli(args):
    parser = argparse.ArgumentParser(description="Override default arguments")
    parser.add_argument("--dataset_dir", type=str, help="Dataset directory")
    parser.add_argument("--lit_dataset_dir", type=str, help="Literal Dataset directory")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--embedding_dim", type=int, help="Embedding dimensions")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--optimize_with_literals", type=bool, help="Optimize with literals"
    )
    parser.add_argument("--lit_lr", type=float, help="Learning rate for literals")
    parser.add_argument(
        "--save_embeddings_as_csv", type=bool, help="Save embeddings as CSV"
    )
    parser.add_argument(
        "--save_experiment", type=bool, help="Save Experiement  Configs and Results"
    )
    cli_args = parser.parse_args()

    # Update only the provided CLI arguments
    for key, value in vars(cli_args).items():
        if value is not None:  # Only overwrite if a new value is provided
            setattr(args, key, value)

    return args


def main(args):
    # Save Experiment Results
    exp_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    exp_path_name = f"Experiments/{exp_date_time}"
    args.full_storage_path = exp_path_name
    os.makedirs(exp_path_name, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and Dataset Initialization
    entity_dataset = read_or_load_kg(args, KG)
    args.num_entities = entity_dataset.num_entities
    args.num_relations = entity_dataset.num_relations

    train_dataset = KvsAll(
        train_set_idx=entity_dataset.train_set,
        entity_idxs=entity_dataset.entity_to_idx,
        relation_idxs=entity_dataset.relation_to_idx,
        form="EntityPrediction",
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    model = Keci(
        args={
            "num_entities": entity_dataset.num_entities,
            "num_relations": entity_dataset.num_relations,
            "embedding_dim": args.embedding_dim,
            "p": 0,
            "q": 1,
            "optim": "Adam",
        }
    ).to(device)

    literal_dataset = None
    Literal_model = None

    if args.optimize_with_literals:
        from src.dataset import LiteralData
        from src.model import LiteralEmbeddings

        literal_dataset = LiteralData(
            dataset_dir=args.lit_dataset_dir, ent_idx=entity_dataset.entity_to_idx
        )
        Literal_model = LiteralEmbeddings(
            num_of_data_properties=literal_dataset.num_data_properties,
            embedding_dims=args.embedding_dim,
        ).to(device)

    # Training
    loss_log = train_model(
        model,
        train_dataloader,
        args,
        device,
        literal_dataset,
        Literal_model,
    )

    # Evaluating the model
    evaluator = Evaluator(args=args)
    model.to("cpu")
    evaluator.eval(
        dataset=entity_dataset,
        trained_model=model,
        form_of_labelling="EntityPrediction",
    )
    model.to(device)
    if args.optimize_with_literals:
        lit_results = evaluate_lit_preds(
            literal_dataset,
            dataset_type="test",
            model=model,
            literal_model=Literal_model,
            device=device,
        )

        print("Training Literal model After Combined Entity-Literal Training")
        Lit_model = train_literal_model(
            args=args, literal_dataset=literal_dataset, kge_model=model, device=device
        )
        print(" Perfromance of Literal Model on Enhanced Entitiy Embeddings ")
        lit_results = evaluate_lit_preds(
            literal_dataset,
            dataset_type="test",
            model=model,
            literal_model=Lit_model,
            device=device,
        )

    if args.save_experiment:
        store(
            trained_model=model,
            model_name="model",
            full_storage_path=args.full_storage_path,
            save_embeddings_as_csv=args.save_embeddings_as_csv,
        )

        print(f"The experiment results are stored at {exp_path_name}")

        df_loss_log = pd.DataFrame.from_dict(loss_log, orient="index").transpose()
        df_loss_log.to_csv(
            os.path.join(exp_path_name, "loss_log.tsv"), sep="\t", index=False
        )

        exp_configs = vars(args)

        with open(os.path.join(exp_path_name, "configs.json"), "w") as f:
            json.dump(exp_configs, f, indent=4)

        with open(os.path.join(exp_path_name, "lp_results.json"), "w") as f:
            json.dump(evaluator.report, f, indent=4)

        lit_results_file_path = os.path.join(exp_path_name, "lit_results.json")
        with open(lit_results_file_path, "w") as f:
            json.dump(lit_results.to_dict(orient="records"), f, indent=4)


if __name__ == "__main__":
    args = update_args_with_cli(args)  # Get command-line arguments
    main(args)  # Pass to main function
