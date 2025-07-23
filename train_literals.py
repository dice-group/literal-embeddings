### Training with Literals
import os

import torch
from torch.utils.data import DataLoader

from src.dataset import LiteralDataset
from src.model import LiteralEmbeddings
from src.static_funcs import (evaluate_lit_preds, load_model_components,
                              save_literal_experiments, train_literal_n_runs)
from src.trainer_literal import train_literal_model


def train_literals(args):
    """Train literal embeddings using a pre-trained KGE model."""
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
    literal_batch_loader = DataLoader(
        literal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=7,
    )
    
    if args.num_literal_runs > 1:
        lit_loss, lit_results = train_literal_n_runs(
            args=args, kge_model=kge_model, literal_dataset=literal_dataset
        )
    else:
        literal_model = LiteralEmbeddings(
            num_of_data_properties=literal_dataset.num_data_properties,
            embedding_dims=kge_model.embedding_dim,
            entity_embeddings=kge_model.entity_embeddings,
            freeze_entity_embeddings=True,
            gate_residual=True
        )
        literal_model, lit_loss = train_literal_model(
            args=args,
            literal_dataset=literal_dataset,
            kge_model=kge_model,
            literal_model=literal_model,
            literal_batch_loader=literal_batch_loader
        )

        if not args.skip_eval_literals:
            lit_results = evaluate_lit_preds(
                literal_dataset,
                dataset_type="test",
                model=kge_model,
                literal_model=literal_model,
                device=args.device,
                multi_regression=args.multi_regression,
            )

    if args.save_experiment:
        save_literal_experiments(args=args, loss_df=lit_loss, results_df=lit_results)
