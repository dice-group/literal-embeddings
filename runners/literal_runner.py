import os
import torch

from src.trainer_literal import train_literal_model
from src.static_funcs import evaluate_lit_preds, load_model_components

from src.static_funcs_literals import apply_best_config_to_args, get_full_storage_path_literals
from src.static_funcs_literals import reset_random_seeds, clear_cuda_cache
from src.static_funcs_literals import  get_literal_datasets, get_litem_model, save_literal_experiments

def train_literals(args):
    """Train literal embeddings using a pre-trained KGE model."""

    # Initial cleanup and  Set initial random seeds
    clear_cuda_cache()
    reset_random_seeds(args.random_seed)

    # Apply best configuration if available
    if args.use_best_config:
        args = apply_best_config_to_args(args)

    # Normalize to lowercase
    args.literal_model = args.literal_model.lower()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # Load and Access components using the dataclass attributes
    model_components = load_model_components(args.pretrained_kge_path)
    kge_model = model_components.model
    total_params = sum(p.numel() for p in kge_model.parameters())
    trainable_params = sum(p.numel() for p in kge_model.parameters() if p.requires_grad)
    e2idx = model_components.entity_to_idx
    args.embedding_dim = model_components.embedding_dim
    args.model = model_components.model_name
    dataset_name = os.path.basename(args.dataset_dir)
    
    print(f"Training Literal Embedding model using pre-trained KGE model at {args.pretrained_kge_path}")
    print(f"Using literal model type: {args.literal_model}")
    print(
        f"KGE parameters: trainable={trainable_params:,}, total={total_params:,}"
    )

    args.full_storage_path = get_full_storage_path_literals(args,dataset_name)
    literal_dataset, literal_dataloader = get_literal_datasets(args, e2idx)

    lit_results = []
    lit_losses = []

    for run in range(args.num_literal_runs):
        print(f"Starting literal training run {run + 1}/{args.num_literal_runs}")
        # Set different seeds per run if needed
        lit_model = get_litem_model(args, literal_dataset, kge_model, run)

        lit_model, lit_loss = train_literal_model( args=args, kge_model=kge_model,
            literal_model=lit_model, literal_batch_loader=literal_dataloader )
        lit_losses.append(lit_loss)

        if not args.skip_eval_literals:
            lit_result = evaluate_lit_preds( literal_dataset, dataset_type="test",
                model=kge_model, literal_model=lit_model,device=args.device )
            lit_results.append(lit_result)

    if args.save_experiment:
        save_literal_experiments(args=args, literal_model=lit_model,
                                attr_to_idx=literal_dataset.data_property_to_idx,
                                lit_results=lit_results, lit_losses=lit_losses)
