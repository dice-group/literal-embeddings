import torch
from dicee.static_funcs import intialize_model
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import \
    StochasticWeightAveraging as SWA
from torch.utils.data import DataLoader

from src.callbacks import (ASWA, EpochLevelProgressBar,
                           PeriodicEvalCallback)
from src.clifford import Lit_Keci
from src.dataset import KvsAll, LiteralDataset, OnevsAllDataset
from src.model import LiteralEmbeddingsCliffordExt, LiteralEmbeddingsExt


def get_callbacks(args):
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
    
    if args.eval_every_n_epochs > 0 or args.eval_at_epochs is not None:
        callbacks.append(PeriodicEvalCallback(experiment_path=args.full_storage_path, max_epochs=args.num_epochs,
                        eval_every_n_epoch=args.eval_every_n_epochs, eval_at_epochs=args.eval_at_epochs,
                        save_model_every_n_epoch=args.save_every_n_epochs, n_epochs_eval_model=args.n_epochs_eval_model))
    return callbacks


def kvsall_collate_fn(batch):
    xs, y_vecs, targets = zip(*batch)  # unzip the triples
    # 1. Stack xs (assuming tensors of same shape)
    xs = torch.stack(xs)
    # 2. Stack one-hot vectors
    y_vecs = torch.stack(y_vecs)
    # 3. Flatten variable-length indices into one long tensor
    targets = [torch.as_tensor(t, dtype=torch.long) for t in targets]
    targets = torch.cat(targets, dim=0)

    return xs, y_vecs, targets


def get_dataloaders(args, entity_dataset):
    train_dataloader = None
    valid_dataloader = None
    train_dataset = None
    valid_dataset = None

    # Prepare training dataloader
    if args.scoring_technique == "KvsAll":
        train_dataset = KvsAll( train_set_idx=entity_dataset.train_set,
                                entity_idxs=entity_dataset.entity_to_idx,
                                  collate_fn = kvsall_collate_fn)
        
        if args.log_validation and not args.train_all_triples:
            valid_dataset = KvsAll(train_set_idx=entity_dataset.valid_set,
                entity_idxs=entity_dataset.entity_to_idx)
    
    elif args.scoring_technique == "1vsAll":
        train_dataset = OnevsAllDataset( train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx, collate_fn = None)
        
        if args.log_validation and not args.train_all_triples:
            valid_dataset = OnevsAllDataset( train_set_idx=entity_dataset.valid_set,
                                            entity_idxs=entity_dataset.entity_to_idx)
    else:
        raise ValueError(f"Unknown scoring technique: {args.scoring_technique}. \
                         Supported techniques: 'KvsAll', 'OneVsAll'")
    
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_core,
        pin_memory=True,        # faster transfer to GPU
        prefetch_factor=2,      # workers prefetch batches
        persistent_workers=True, # workers stay alive across epochs
        collate_fn= train_dataset.collate_fn
    )

    if valid_dataset is not None:
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                       shuffle=False, num_workers=7 )
        
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
            gate_residual=getattr(args, 'gate_residual', False),
            freeze_entity_embeddings=args.freeze_entity_embeddings_combined,
        )
    else:
        raise ValueError(f"Unknown literal model: {args.literal_model}.\
                          Supported models: 'clifford', 'mlp'")
    return literal_dataset, Literal_model


def get_model(args, entity_dataset = None):
    if args.model == "Lit_Keci":
        kge_model = Lit_Keci(args=vars(args), ent2idx=entity_dataset.entity_to_idx,
                              rel2idx=entity_dataset.relation_to_idx)
    else:
        kge_model, _ = intialize_model(vars(args), 0)
    return kge_model