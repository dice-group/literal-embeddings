### Main File
import json
import os
from datetime import datetime

import pandas as pd
import torch
from dicee.dataset_classes import KvsAll
from dicee.evaluator import Evaluator
from dicee.knowledge_graph import KG
from dicee.static_funcs import intialize_model, read_or_load_kg, store
from torch.utils.data import DataLoader

from src.config import get_default_arguments
from src.dataset import LiteralData
from src.model import LiteralEmbeddings
from src.static_funcs import (save_kge_experiments, save_literal_experiments,
                              train_literal_n_runs)
from src.trainer import train_literal_model, train_model
from src.utils import evaluate_lit_preds, load_model_components


def main(args):
    # Save Experiment Results
    if args.full_storage_path is None:
        exp_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        exp_path_name = f"Test_runs/{exp_date_time}"
        args.full_storage_path = exp_path_name
    os.makedirs(args.full_storage_path, exist_ok=True)

    # Device setup
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # Model and Dataset Initialization
    entity_dataset = read_or_load_kg(args, KG)
    args.num_entities = entity_dataset.num_entities
    args.num_relations = entity_dataset.num_relations

    # kvsall dataset initialization
    train_dataset = KvsAll(
        train_set_idx=entity_dataset.train_set,
        entity_idxs=entity_dataset.entity_to_idx,
        relation_idxs=entity_dataset.relation_to_idx,
        form="EntityPrediction",
        label_smoothing_rate=args.label_smoothing_rate,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # initialize KGE model using dicee framework model initialization
    kge_model, _ = intialize_model(vars(args), 0)
    literal_dataset = None
    Literal_model = None
    lit_results = None

    if args.combined_training:
        # intialize literal embedding model and dataset for combined training with KGE
        literal_dataset = LiteralData(
            dataset_dir=args.dataset_dir,
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

    # Evaluating the KGE model on Link Prediction task MRR, H@1,3,10
    evaluator = Evaluator(args=args)
    kge_model.to("cpu")
    evaluator.eval(
        dataset=entity_dataset,
        trained_model=kge_model,
        form_of_labelling="EntityPrediction",
    )

    # can be used to skip literal eval if all literal data is used as train
    # can be used also if no test/val split avilable
    # use args.eval_literals = Flase to skip this step
    if args.combined_training and args.eval_literals:
        lit_results = evaluate_lit_preds(
            literal_dataset,
            dataset_type="test",
            model=kge_model,
            literal_model=Literal_model,
            device=args.device,
            multi_regression=args.multi_regression,
        )

    # save kge model, training configs,
    if args.save_experiment:
        store(
            trained_model=kge_model,
            model_name="model",
            full_storage_path=args.full_storage_path,
            save_embeddings_as_csv=args.save_embeddings_as_csv,
        )
        save_kge_experiments(args=args, loss_log=loss_log, lit_results=lit_results)


def train_with_kge(args):

    # torch related initializations
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # load KGE model as dicee KGE object or local KGE object
    kge_model, configs, e2idx, _ = load_model_components(args.pretrained_kge_path)
    args.embedding_dim = kge_model.embedding_dim
    args.model = kge_model.name
    args.dataset_dir = configs["dataset_dir"]
    dataset_name = os.path.basename(args.dataset_dir)

    print(
        "Training Literal Embedding model using pre-trained KGE model at %s"
        % args.pretrained_kge_path
    )

    if not args.full_storage_path:
        args.full_storage_path = (
            f"Experiments_Literals/{dataset_name}_{args.embedding_dim}/{args.model}"
        )

    literal_dataset = LiteralData(
        dataset_dir=args.dataset_dir, ent_idx=e2idx, normalization=args.lit_norm
    )
    if args.num_literal_runs > 1:
        lit_loss, lit_results = train_literal_n_runs(
            args=args, kge_model=kge_model, literal_dataset=literal_dataset
        )

    else:
        Literal_model = LiteralEmbeddings(
            num_of_data_properties=literal_dataset.num_data_properties,
            embedding_dims=kge_model.embedding_dim,
            multi_regression=args.multi_regression,
        )

        Literal_model, lit_loss = train_literal_model(
            args=args,
            literal_dataset=literal_dataset,
            kge_model=kge_model,
            Literal_model=Literal_model,
        )

        if args.eval_literals:
            lit_results = evaluate_lit_preds(
                literal_dataset,
                dataset_type="test",
                model=kge_model,
                literal_model=Literal_model,
                device=args.device,
                multi_regression=args.multi_regression,
            )

    if args.save_experiment:
        save_literal_experiments(args=args, loss_df=lit_loss, results_df=lit_results)


if __name__ == "__main__":
    args = get_default_arguments()
    args.learning_rate = args.lr
    if args.literal_training:
        train_with_kge(args)
    else:
        main(args)  # Pass to main function
