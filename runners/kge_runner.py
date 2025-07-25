### KGE Training Runner
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

from src.abstracts import KGETrainer
from src.callbacks import ASWA, EpochLevelProgressBar
from src.dataset import LiteralDataset
from src.model import LiteralEmbeddings
from src.static_funcs import evaluate_lit_preds, save_kge_experiments
from src.trainer import KGE_Literal


def train_kge_model(args):
    """Train a KGE model with optional literal embeddings."""
    # Set up experiment storage path
    if not args.full_storage_path:
        dataset_name = os.path.basename(args.dataset_dir)
        if not dataset_name:
            dataset_name = os.path.basename(args.path_single_kg)
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

    kge_model, _ = intialize_model(vars(args), 0)

    # Evaluator and Lightning module
    evaluator = Evaluator(args=args)
    kge_model_lightning = KGE_Literal(
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
