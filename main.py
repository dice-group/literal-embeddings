### Main File
import argparse
import json
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.config import get_default_arguments
from dicee.dataset_classes import KvsAll
from dicee.evaluator import Evaluator
from dicee.knowledge_graph import KG
from dicee.knowledge_graph_embeddings import KGE
from dicee.models import Keci, TransE
from dicee.static_funcs import intialize_model, read_or_load_kg, store
from torch.utils.data import DataLoader

from src.dataset import LiteralData
from src.model import LiteralEmbeddings
from src.trainer import train_literal_model, train_model
from src.utils import evaluate_lit_preds

def main(args):
    # Save Experiment Results
    if args.full_storage_path is None:
        exp_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        exp_path_name = f"Test_runs/{exp_date_time}"
        args.full_storage_path = exp_path_name
    os.makedirs(args.full_storage_path, exist_ok=True)

    # Device setup
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    kge_model, _ = intialize_model(vars(args), 0)
    literal_dataset = None
    Literal_model = None

    if args.combined_training:

        literal_dataset = LiteralData(
            dataset_dir=args.lit_dataset_dir,
            ent_idx=entity_dataset.entity_to_idx,
            normalization=args.lit_norm,
        )
        Literal_model = LiteralEmbeddings(
            num_of_data_properties=literal_dataset.num_data_properties,
            embedding_dims=args.embedding_dim,
            multi_regression=args.multi_regression,
        )

    # Training
    kge_model, Literal_model, loss_log = train_model(
        kge_model,
        train_dataloader,
        args,
        literal_dataset,
        Literal_model,
    )

    # Evaluating the model
    evaluator = Evaluator(args=args)
    kge_model.to("cpu")
    evaluator.eval(
        dataset=entity_dataset,
        trained_model=kge_model,
        form_of_labelling="EntityPrediction",
    )
    if args.combined_training:
        lit_results = evaluate_lit_preds(
            literal_dataset,
            dataset_type="test",
            model=kge_model,
            literal_model=Literal_model,
            device=args.device,
            multi_regression=args.multi_regression,
        )

        print("Training Literal model After Combined Entity-Literal Training")
        Lit_model, _ = train_literal_model(
            args=args,
            literal_dataset=literal_dataset,
            kge_model=kge_model,
        )
        print(" Perfromance of Literal Model on Enhanced Entitiy Embeddings ")
        lit_results_enhanced = evaluate_lit_preds(
            literal_dataset,
            dataset_type="test",
            model=kge_model,
            literal_model=Lit_model,
            device=args.device,
            multi_regression=args.multi_regression,
        )
        if args.save_experiment:
            lit_results_file_path = os.path.join(exp_path_name, "lit_results.json")
            with open(lit_results_file_path, "w") as f:
                json.dump(lit_results.to_dict(orient="records"), f, indent=4)

    if args.save_experiment:
        store(
            trained_model=kge_model,
            model_name="model",
            full_storage_path=args.full_storage_path,
            save_embeddings_as_csv=args.save_embeddings_as_csv,
        )

        print(f"The experiment results are stored at {exp_path_name}")

        df_loss_log = pd.DataFrame.from_dict(loss_log, orient="index").transpose()
        df_loss_log.to_csv(
            os.path.join(exp_path_name, "loss_log.tsv"), sep="\t", index=False
        )
        args.device = str(args.device)
        exp_configs = vars(args)

        with open(os.path.join(exp_path_name, "configuration.json"), "w") as f:
            json.dump(exp_configs, f, indent=4)

        with open(os.path.join(exp_path_name, "lp_results.json"), "w") as f:
            json.dump(evaluator.report, f, indent=4)


def train_with_kge(args):

    print(
        "Training Literal Embedding model using pre-trained KGE model at %s"
        % args.pretrained_kge_path
    )
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        config_path = os.path.join(args.pretrained_kge_path, "configuration.json")
        model_path = os.path.join(args.pretrained_kge_path, "model.pt")
        entity_to_idx_path = os.path.join(args.pretrained_kge_path, "entity_to_idx.csv")

        # Load configuration
        with open(config_path) as json_file:
            configs = json.load(json_file)

        # Load model weights
        weights = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=True
        )

        # Initialize the model
        kge_model, _ = intialize_model(configs, 0)

        # Load the model weights into the model
        kge_model.load_state_dict(weights)
        e2idx_df = pd.read_csv(entity_to_idx_path, index_col=0)

    except:
        print(" Building the KGE model failed: Fix args ")
        exit(0)

    literal_dataset = LiteralData(
        dataset_dir=args.lit_dataset_dir, ent_idx=e2idx_df, normalization=args.lit_norm
    )
    Lit_model, loss_log = train_literal_model(
        args=args,
        literal_dataset=literal_dataset,
        kge_model=kge_model,
    )

    lit_results = evaluate_lit_preds(
        literal_dataset,
        dataset_type="test",
        model=kge_model,
        literal_model=Lit_model,
        device=args.device,
        multi_regression=args.multi_regression,
    )
    if args.save_experiment:
        args.device = str(args.device)
        exp_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        exp_path_name = f"Experiments/{exp_date_time}"
        os.makedirs(exp_path_name, exist_ok=True)
        lit_results_file_path = os.path.join(exp_path_name, "lit_results.json")
        with open(lit_results_file_path, "w") as f:
            json.dump(lit_results.to_dict(orient="records"), f, indent=4)
        with open(os.path.join(exp_path_name, "configuration.json"), "w") as f:
            json.dump(configs, f, indent=4)

        df_loss_log = pd.DataFrame.from_dict(loss_log, orient="index").transpose()
        df_loss_log.to_csv(
            os.path.join(exp_path_name, "loss_log.tsv"), sep="\t", index=False
        )


if __name__ == "__main__":
    args = get_default_arguments()
    args.learning_rate = args.lr
    if args.literal_training:
        train_with_kge(args)
    else:
        main(args)  # Pass to main function
