### KGE Training Runner
import os
import torch

from dicee.evaluator import Evaluator
from dicee.knowledge_graph import KG
from dicee.static_funcs import  read_or_load_kg, store

from src.abstracts import KGETrainer
from src.trainer import KGE_Literal
from src.static_funcs import evaluate_lit_preds, save_kge_experiments, get_full_storage_path
from src.static_train_utils import get_callbacks, get_dataloaders, get_literal_components, get_model

def train_kge_model(args):
    """Train a KGE model with optional literal embeddings."""
    # Set up experiment storage path
    args.learning_rate = args.lr
    args.full_storage_path = get_full_storage_path(args)
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

    train_dataloader, valid_dataloader = get_dataloaders(args, entity_dataset)

    kge_model = get_model(args = args, entity_dataset=entity_dataset)
    literal_dataset, Literal_model = None, None

    # Combined training and literal model setup
    if args.combined_training:
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
        logger=False,
        check_val_every_n_epoch=check_val_epochs,
        log_every_n_steps=0,
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
