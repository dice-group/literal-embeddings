### Main File
import os
from datetime import datetime

import torch
from dicee.dataset_classes import KvsAll
from dicee.evaluator import Evaluator
from dicee.knowledge_graph import KG
from dicee.static_funcs import intialize_model, read_or_load_kg, store
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import \
    StochasticWeightAveraging as SWA
from torch.utils.data import DataLoader

from src.literalE_models import DistMult_EA
from src.abstracts import KGETrainer
from src.callbacks import ASWA, EpochLevelProgressBar
from src.config import get_default_arguments
from src.dataset import LiteralDataset
from src.model import LiteralEmbeddings
from src.static_funcs import (evaluate_lit_preds, load_model_components,
                              save_kge_experiments, save_literal_experiments,
                              train_literal_n_runs)
from src.trainer import KGEModelLightning
from src.trainer_literal import train_literal_model


def main(args):
    # Set up experiment storage path
    if not args.full_storage_path:
        dataset_name = os.path.basename(args.dataset_dir)
        if not dataset_name:
            dataset_name  = os.path.basename(args.path_single_kg)
        if args.test_runs:
            exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            args.full_storage_path = f"Test_runs/{exp_time}"
        elif args.combined_training:
            args.full_storage_path = f"Experiments/KGE_Combined/{dataset_name}_combined/{args.model}"
        else:
            args.full_storage_path = f"Experiments/KGE/{dataset_name}/{args.model}"
    os.makedirs(args.full_storage_path, exist_ok=True)
    print("Training dir", args.full_storage_path)

    # Device and seed setup
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # Load dataset and set entity/relation counts
    entity_dataset = read_or_load_kg(args, KG)
    args.num_entities = entity_dataset.num_entities
    args.num_relations = entity_dataset.num_relations

    # Prepare training dataloader
    train_dataset = KvsAll(
        train_set_idx=entity_dataset.train_set,
        entity_idxs=entity_dataset.entity_to_idx,
        relation_idxs=entity_dataset.relation_to_idx,
        form="EntityPrediction",
        label_smoothing_rate=args.label_smoothing_rate,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=7
    )

    # Prepare validation dataloader if needed
    valid_dataloader = None
    if args.log_validation and not args.train_all_triples:
        valid_dataset = KvsAll(
            train_set_idx=entity_dataset.valid_set,
            entity_idxs=entity_dataset.entity_to_idx,
            relation_idxs=entity_dataset.relation_to_idx,
            form="EntityPrediction",
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7
        )

    # Combined training and literal model setup
    literal_dataset = None
    Literal_model = None
    if args.model == "DistMult_EA":
        args.combined_training = True
        args.skip_eval_literals = True
    if args.combined_training:
        literal_dataset = LiteralDataset(
            dataset_dir=args.dataset_dir,
            ent_idx=entity_dataset.entity_to_idx,
            normalization=args.lit_norm,
        )
        args.num_attributes = literal_dataset.num_data_properties
        Literal_model = LiteralEmbeddings(
            num_of_data_properties=literal_dataset.num_data_properties,
            embedding_dims=args.embedding_dim,
            multi_regression=args.multi_regression,
        )

    # Model initialization
    if args.model == "DistMult_EA":
        kge_model = DistMult_EA(vars(args))
    else:
        kge_model, _ = intialize_model(vars(args), 0)

    # Evaluator and Lightning module
    evaluator = Evaluator(args=args)
    kge_model_lightning = KGEModelLightning(
        kge_model, Literal_model, args, literal_dataset
    )

    # Callbacks setup
    callbacks = [EpochLevelProgressBar()]
    if args.early_stopping:
        callbacks.append(EarlyStopping(
            monitor="ent_loss_val",
            patience=args.early_stopping_patience,
            mode="min",
        ))
    if args.swa:
        callbacks.append(SWA(swa_epoch_start=1, swa_lrs=0.05))
    elif args.adaptive_swa:
        callbacks.append(ASWA(num_epochs=args.num_epochs, path=args.full_storage_path))
    else:
        print("No Stochastic Weight Averaging (SWA) or Adaptive SWA (ASWA) used.")

    # Trainer setup
    trainer = KGETrainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=False,
        check_val_every_n_epoch=0,
        log_every_n_steps=0,
        enable_checkpointing=False,
        evaluator=evaluator,
        entity_dataset=entity_dataset,
    )

    # Training
    trainer.fit(kge_model_lightning, train_dataloader, valid_dataloader)

    # Literal evaluation if needed
    lit_results = None
    if args.combined_training and not args.skip_eval_literals:
        lit_results = evaluate_lit_preds(
            literal_dataset,
            dataset_type="test",
            model=kge_model,
            literal_model=Literal_model,
            device=args.device,
            multi_regression=args.multi_regression,
        )

    # Save experiment results
    if args.save_experiment:
        store(
            trained_model=kge_model,
            model_name="model",
            full_storage_path=args.full_storage_path,
            save_embeddings_as_csv=args.save_embeddings_as_csv,
        )
        save_kge_experiments(args=args, loss_log={}, lit_results=lit_results)
        print(
            f"Experiment for {args.model} + {args.embedding_dim} (combined={args.combined_training}) completed and stored at {args.full_storage_path}"
        )

def train_with_kge(args):

    # torch related initializations
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # load KGE model as dicee KGE object or local KGE object
    
    model_components = load_model_components(args.pretrained_kge_path)

    if model_components is None:
        print("Failed to load model components.")
        return  # Or handle as needed (e.g., raise Exception)

    kge_model, configs, e2idx, _ = model_components
    
    args.embedding_dim = kge_model.embedding_dim
    args.model = kge_model.name
    #args.dataset_dir = configs["dataset_dir"]
    dataset_name = os.path.basename(args.dataset_dir)

    print(
        "Training Literal Embedding model using pre-trained KGE model at %s"
        % args.pretrained_kge_path
    )

    if not args.full_storage_path:
        args.full_storage_path = (
            f"Experiments/Literals/{dataset_name}/{args.model}_{args.embedding_dim}"
        )

    literal_dataset = LiteralDataset(
        dataset_dir=args.dataset_dir, ent_idx=e2idx, normalization=args.lit_norm, sampling_ratio=args.lit_sampling_ratio
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

        if not args.skip_eval_literals:
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
# This script is the main entry point for training and evaluating a KGE model with literal embeddings.