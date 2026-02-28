from src.callbacks import ASWA, EpochLevelProgressBar, PeriodicEvalCallback, BatchProcessCallback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import \
    StochasticWeightAveraging as SWA
import torch
from src.dataset import LiteralDataset, KvsAll, OnevsAllDataset
from torch.utils.data import DataLoader
from src.model import LiteralEmbeddingsExt, LiteralEmbeddingsCliffordExt
from src.clifford import Lit_Keci
from dicee.static_funcs import  intialize_model

def get_callbacks(args):
    # Callbacks setup
    callbacks = [EpochLevelProgressBar()]
    # if args.literalE:
    #     callbacks.append(LiteralCallback(args))
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
    
    if args.eval_every_n_epochs > 0 or args.eval_at_epochs is not None:
        callbacks.append(PeriodicEvalCallback(experiment_path=args.full_storage_path, max_epochs=args.num_epochs,
                        eval_every_n_epoch=args.eval_every_n_epochs, eval_at_epochs=args.eval_at_epochs,
                        save_model_every_n_epoch=args.save_every_n_epochs, n_epochs_eval_model=args.n_epochs_eval_model))
    return callbacks


def collate_fn(batch):
    xs, y_vecs, targets = zip(*batch)  # unzip the triples
    # 1. Stack xs (assuming tensors of same shape)
    xs = torch.stack(xs)
    # 2. Stack one-hot vectors
    y_vecs = torch.stack(y_vecs)
    # 3. Flatten variable-length indices into one long tensor
    #    Convert each target to tensor if it's not already
    if isinstance(targets, tuple) and isinstance(targets[0], list):
        targets = [torch.as_tensor(t, dtype=torch.long) for t in targets]
        targets = torch.cat(targets, dim=0)
        
    else:
        targets = torch.stack(targets)

    return xs, y_vecs, targets

def get_dataloaders(args, entity_dataset):
    train_dataloader = None
    valid_dataloader = None
    train_dataset = None
    valid_dataset = None

    # Prepare training dataloader
    if args.scoring_technique == "KvsAll":
        train_dataset = KvsAll(
            train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx,
        )
        if args.log_validation and not args.train_all_triples:
            valid_dataset = KvsAll(
                train_set_idx=entity_dataset.valid_set,
                entity_idxs=entity_dataset.entity_to_idx,
        )
    elif args.scoring_technique == "1vsAll":
        train_dataset = OnevsAllDataset(
            train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx
        )
        if args.log_validation and not args.train_all_triples:
            valid_dataset = OnevsAllDataset(
                train_set_idx=entity_dataset.valid_set,
                entity_idxs=entity_dataset.entity_to_idx
            )
    else:
        raise ValueError(f"Unknown scoring technique: {args.scoring_technique}. Supported techniques: 'KvsAll', 'OneVsAll'")
    
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_core,
        pin_memory=True,        # faster transfer to GPU
        prefetch_factor=2,      # workers prefetch batches
        persistent_workers=True, # workers stay alive across epochs
        collate_fn=collate_fn  # custom collate function
    )

    if valid_dataset is not None:
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7
        )


    return train_dataloader, valid_dataloader


def get_literal_components(args, entity_dataset):
    literal_dataset = LiteralDataset(
            dataset_dir=args.dataset_dir,
            ent_idx=entity_dataset.entity_to_idx,
            normalization=args.lit_norm,
        )
    args.num_attributes = literal_dataset.num_data_properties
        
    # Use external embedding model that takes embeddings as input
    if args.literal_model == "clifford":
        Literal_model = LiteralEmbeddingsCliffordExt(
            num_of_data_properties=literal_dataset.num_data_properties,
            embedding_dims=args.embedding_dim,
            dropout=getattr(args, 'dropout', 0.15),
            gate_residual=getattr(args, 'gate_residual', False),
            freeze_entity_embeddings=args.freeze_entity_embeddings_combined,
        )
    elif args.literal_model == "mlp":
        Literal_model = LiteralEmbeddingsExt(
            num_of_data_properties=literal_dataset.num_data_properties,
            embedding_dims=args.embedding_dim,
            dropout=getattr(args, 'dropout', 0.3),
            gate_residual=False,
            freeze_entity_embeddings=args.freeze_entity_embeddings_combined,
        )
    else:
        raise ValueError(f"Unknown literal model: {args.literal_model}. Supported models: 'clifford', 'mlp'")
    return literal_dataset, Literal_model


def get_model(args, entity_dataset = None):
    if args.model == "Lit_Keci":
        kge_model = Lit_Keci(args=vars(args), ent2idx=entity_dataset.entity_to_idx, rel2idx=entity_dataset.relation_to_idx)
    else:
        kge_model, _ = intialize_model(vars(args), 0)
    return kge_model


def get_entity_embedding_weight(kge_model):
    if not hasattr(kge_model, "entity_embeddings"):
        return None
    entity_embeddings = kge_model.entity_embeddings
    if hasattr(entity_embeddings, "weight"):
        return entity_embeddings.weight
    return None


def grad_wrt_entity_weights(kge_model, loss: torch.Tensor, retain_graph: bool = True):
    ent_weight = get_entity_embedding_weight(kge_model)
    if ent_weight is None:
        return None
    grad = torch.autograd.grad(
        outputs=loss,
        inputs=ent_weight,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )[0]
    return grad


def compute_dynamic_lit_weight(
    kge_model,
    device,
    ent_loss,
    lit_loss,
    entity_ids,
    target_ratio: float = 1.0,
    min_w: float = 1e-4,
    max_w: float = 10.0,
    eps: float = 1e-12,
):
    """Compute lambda_lit from gradient norms on entity embeddings."""
    if entity_ids is None or entity_ids.numel() == 0:
        return torch.tensor(1.0, device=device)

    tracked_ids = torch.unique(entity_ids.long()).to(device)
    ent_grad_full = grad_wrt_entity_weights(kge_model, ent_loss, retain_graph=True)
    lit_grad_full = grad_wrt_entity_weights(kge_model, lit_loss, retain_graph=True)

    if ent_grad_full is None:
        return torch.tensor(1.0, device=device)

    ent_grad = ent_grad_full.index_select(0, tracked_ids)
    if lit_grad_full is None:
        lit_grad = torch.zeros_like(ent_grad)
    else:
        lit_grad = lit_grad_full.index_select(0, tracked_ids)

    ent_norm = torch.linalg.vector_norm(ent_grad)
    lit_norm = torch.linalg.vector_norm(lit_grad)
    lambda_lit = target_ratio * (ent_norm / (lit_norm + eps))
    lambda_lit = torch.clamp(lambda_lit, min=min_w, max=max_w).detach()
    return lambda_lit
