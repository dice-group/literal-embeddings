### KGE Training Runner
import os
import shutil
import torch

from dicee.evaluator import Evaluator
from dicee.knowledge_graph import KG
from dicee.static_funcs import  read_or_load_kg, store

from src.abstracts import KGETrainer
from src.kbln import attach_kbln_model
from src.literale import attach_literale_embeddings, get_literal_dataset
from src.trainer import KGE_Literal
from src.static_funcs import evaluate_lit_preds, save_kge_experiments, get_full_storage_path
from src.static_train_utils import (
    get_callbacks,
    get_dataloaders,
    get_literal_components,
    get_model,
)


def prepare_experiment_dir(path):
    if os.path.isdir(path) and os.listdir(path):
        print(f"Removing existing experiment directory: {path}")
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)

def train_kge_model(args):
    """Train a KGE model with optional literal embeddings."""
    if sum(bool(mode) for mode in (args.literalE, args.kbln, args.combined_training)) > 1:
        raise ValueError("`literalE`, `kbln`, and `combined_training` are mutually exclusive.")

    # Set up experiment storage path
    args.learning_rate = args.lr
    args.full_storage_path = get_full_storage_path(args)
    prepare_experiment_dir(args.full_storage_path)
    print("Training dir", args.full_storage_path)

    # Device and seed setup
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    # Load dataset and set entity/relation counts
    entity_dataset = read_or_load_kg(args, KG)
    args.num_entities = entity_dataset.num_entities
    args.num_relations = entity_dataset.num_relations

    train_dataloader, valid_dataloader = get_dataloaders(args, entity_dataset)

    kge_model = get_model(args = args, entity_dataset=entity_dataset)
    literal_dataset, Literal_model = None, None

    if args.literalE:
        literal_dataset = get_literal_dataset(args, entity_dataset)
        kge_model = attach_literale_embeddings(kge_model, literal_dataset)
    elif args.kbln:
        literal_dataset = get_literal_dataset(args, entity_dataset)
        kge_model, literal_dataset = attach_kbln_model(args, kge_model, entity_dataset, literal_dataset=literal_dataset)
    elif args.combined_training:
        literal_dataset, Literal_model = get_literal_components(args, entity_dataset)

    # Evaluator and Lightning module
    evaluator = Evaluator(args=args)
    lightning_module = KGE_Literal(
        kge_model, Literal_model, args, literal_dataset
    )

    # set validation epochs
    check_val_epochs = 1 if args.log_validation and valid_dataloader else None

    # Trainer setup
    trainer = KGETrainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=1,
        callbacks=get_callbacks(args),
        check_val_every_n_epoch=check_val_epochs,
        log_every_n_steps=1,
        enable_checkpointing=False,
        evaluator=evaluator,
        entity_dataset=entity_dataset,
        deterministic=True
    )

    # Training
    trainer.fit(lightning_module, train_dataloader, valid_dataloader)

    # Literal evaluation if needed
    lit_results = None
    if args.combined_training and not args.skip_eval_literals:
        lit_results = evaluate_lit_preds( literal_dataset, dataset_type="test", 
         model=lightning_module.kge_model, literal_model=lightning_module.Literal_model, device=args.device )

    # Save experiment results
    if args.save_experiment:
        store(trained_model=lightning_module.kge_model, model_name="model",
            full_storage_path=args.full_storage_path,
            save_embeddings_as_csv=args.save_embeddings_as_csv
        )
        if literal_dataset:
            save_kge_experiments(args=args, loss_log={}, lit_results=lit_results, attr_to_idx=literal_dataset.data_property_to_idx)
        else:
            save_kge_experiments(args=args, loss_log={})
        print( f"Experiment for {args.model} + {args.embedding_dim} (combined={args.combined_training})"
               f"completed and stored at {args.full_storage_path}"
        )
