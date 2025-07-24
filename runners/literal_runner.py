import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import LiteralDataset
from src.model import LiteralEmbeddings, LiteralEmbeddingsClifford
from src.static_funcs import (evaluate_lit_preds, load_model_components,
                              save_literal_experiments, KGEModelComponents)
from src.trainer_literal import train_literal_model
from dicee.static_funcs import save_checkpoint_model
from src.static_funcs import evaluate_link_prediction_performance_with_reciprocals

def train_literals(args):
    """Train literal embeddings using a pre-trained KGE model."""
    
    # Normalize to lowercase
    args.literal_model = args.literal_model.lower()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = True

    model_components = load_model_components(args.pretrained_kge_path)
    if model_components is None:
        print("Failed to load model components.")
        return

    # Access components using the dataclass attributes
    kge_model = model_components.model
    configs = model_components.config
    e2idx = model_components.entity_to_idx
    args.embedding_dim = model_components.embedding_dim
    args.model = model_components.model_name
    dataset_name = os.path.basename(args.dataset_dir)

    print(f"Training Literal Embedding model using pre-trained KGE model at {args.pretrained_kge_path}")
    print(f"Using literal model type: {args.literal_model}")

    if not args.full_storage_path:
        args.full_storage_path = (
            f"Experiments/Literals/{dataset_name}/{args.model}_{args.embedding_dim}_{args.literal_model}"
        )

    literal_dataset = LiteralDataset(
        dataset_dir=args.dataset_dir, ent_idx=e2idx, normalization=args.lit_norm, sampling_ratio=args.lit_sampling_ratio
    )
    literal_batch_loader = DataLoader(
        literal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count()),  # Tune this
        pin_memory=args.device.type == "cuda",
    )

    final_loss_df = None
    final_results_df = None

    for run in range(args.num_literal_runs):
        print(f"Starting literal training run {run + 1}/{args.num_literal_runs}")
        # Set different seeds per run if needed
        torch.manual_seed(args.random_seed + run)
        if args.device.type == "cuda":
            torch.cuda.manual_seed_all(args.random_seed + run)

        # Initialize the appropriate literal model based on args.literal_model
        if args.literal_model == 'clifford':
            literal_model = LiteralEmbeddingsClifford(
                num_of_data_properties=literal_dataset.num_data_properties,
                embedding_dims=kge_model.embedding_dim,
                entity_embeddings=kge_model.entity_embeddings,
                freeze_entity_embeddings=args.freeze_entity_embeddings,
                gate_residual=args.gate_residual,
                dropout=getattr(args, 'dropout', 0.15),
            )
        elif args.literal_model == 'mlp':
            literal_model = LiteralEmbeddings(
                num_of_data_properties=literal_dataset.num_data_properties,
                embedding_dims=kge_model.embedding_dim,
                entity_embeddings=kge_model.entity_embeddings,
                freeze_entity_embeddings=args.freeze_entity_embeddings,
                dropout=getattr(args, 'dropout', 0.3),
                gate_residual=getattr(args, 'gate_residual', True),
            )
            
        else:
            raise ValueError(f"Unknown literal model type: {args.literal_model}. Supported types: 'mlp', 'clifford'")
            
        
        literal_model, lit_loss = train_literal_model(
            args=args,
            kge_model=kge_model,
            literal_model=literal_model,
            literal_batch_loader=literal_batch_loader
        )

        lit_results = None
        if not args.skip_eval_literals:
            lit_results = evaluate_lit_preds(
                literal_dataset,
                dataset_type="test",
                model=kge_model,
                literal_model=literal_model,
                device=args.device,
                multi_regression=args.multi_regression,
            )
            # Convert lit_results to DataFrame and rename columns
            df_results = pd.DataFrame(lit_results)
            df_results.rename(
                columns={
                    col: f"{col}_run_{run+1}" for col in df_results.columns if col != "relation"
                },
                inplace=True,
            )

            # Convert lit_loss to DataFrame and rename
            df_loss = pd.DataFrame(lit_loss)
            df_loss.rename(columns={"lit_loss": f"lit_loss_run_{run+1}"}, inplace=True)

            # Initialize and add epoch index for the first run
            if final_loss_df is None:
                final_loss_df = df_loss.copy()
                final_loss_df.insert(0, "epoch", range(1, len(df_loss) + 1))
                
            else:
                final_loss_df = pd.concat([final_loss_df, df_loss], axis=1)
                
            if final_results_df is None:
                final_results_df = df_results.copy()
            else:
                final_results_df = pd.merge(final_results_df, df_results, on="relation", how="left")



    if args.save_experiment:
        save_literal_experiments(
            args=args,
            literal_model=literal_model,
            loss_df=final_loss_df,
            results_df=final_results_df
        )

    if not args.freeze_entity_embeddings:
        test_dir = os.path.join(args.dataset_dir, "test.txt")
        assert os.path.exists(test_dir), f"Test file not found at {test_dir}"
        test_triples = pd.read_csv(test_dir,
                           sep="\s+",
                           header=None, usecols=[0, 1, 2],
                           names=['subject', 'relation', 'object'],
                           dtype=str).values.tolist()
        lp_results_kge = evaluate_link_prediction_performance_with_reciprocals(kge_model, triples=test_triples,
                                                                            model_components=model_components)
        print(f"Link prediction results before literal training:\n {lp_results_kge}")
        # Save the model with updated entity embeddings
        kge_model.entity_embeddings.weight.data.copy_(literal_model.entity_embeddings.weight.data)

        save_checkpoint_model(
            model=kge_model,
            path =args.full_storage_path + f"/lit_augmented_model.pt",
        )
        
        lp_results_kge_lit = evaluate_link_prediction_performance_with_reciprocals(kge_model, triples=test_triples,
                                                                            model_components=model_components)
        print(f"Link prediction results after literal training:\n {lp_results_kge_lit}")