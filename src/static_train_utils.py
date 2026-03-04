from src.callbacks import ASWA, EpochLevelProgressBar, PeriodicEvalCallback, BatchProcessCallback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import \
    StochasticWeightAveraging as SWA
import torch
from src.dataset import KvsAll, OnevsAllDataset
from torch.utils.data import DataLoader
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
    xs, y_vecs = zip(*batch)
    return torch.stack(xs), torch.stack(y_vecs)

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
            entity_idxs=entity_dataset.entity_to_idx,
        )
        if args.log_validation and not args.train_all_triples:
            valid_dataset = OnevsAllDataset(
                train_set_idx=entity_dataset.valid_set,
                entity_idxs=entity_dataset.entity_to_idx,
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
def get_literal_components(args, entity_dataset, literal_dataset=None):
    if literal_dataset is None:
        from src.literale import get_literal_dataset
        literal_dataset = get_literal_dataset(args, entity_dataset)

    from src.model import LiteralEmbeddingsCliffordExt, LiteralEmbeddingsExt
        
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
        from src.clifford import Lit_Keci
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


def compute_literal_loss_mix(
    kge_model,
    device,
    ent_loss,
    lit_loss,
    base_mix: float = 0.5,
    target_ratio: float = 1.0,
    min_mix: float = 0.0,
    eps: float = 1e-12,
):
    """Compute an effective literal mix from shared entity-embedding gradient norms.

    This implements the archived convex-style formulation:
        s_eff = min(s, tau * g_ent / (g_lit + eps))

    Falls back to the static base mix if no usable shared gradients are available,
    e.g. when entity embeddings are detached for the literal branch.
    """
    ent_grad_full = grad_wrt_entity_weights(kge_model, ent_loss, retain_graph=True)
    lit_grad_full = grad_wrt_entity_weights(kge_model, lit_loss, retain_graph=True)

    fallback_mix = torch.tensor(base_mix, device=device)
    if ent_grad_full is None or lit_grad_full is None:
        return fallback_mix, False

    ent_norm = torch.linalg.vector_norm(ent_grad_full)
    lit_norm = torch.linalg.vector_norm(lit_grad_full)
    if not torch.isfinite(ent_norm) or not torch.isfinite(lit_norm) or lit_norm <= eps:
        return fallback_mix, False

    proposed_mix = target_ratio * (ent_norm / (lit_norm + eps))
    effective_mix = torch.minimum(
        torch.tensor(base_mix, device=device),
        proposed_mix,
    )
    effective_mix = torch.clamp(effective_mix, min=min_mix).detach()
    return effective_mix, True
