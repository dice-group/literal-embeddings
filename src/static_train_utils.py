from src.callbacks import ASWA, EpochLevelProgressBar, PeriodicEvalCallback, BatchProcessCallback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import \
    StochasticWeightAveraging as SWA
from src.dataset import LiteralDataset, KvsAll, OneVsAll
from torch.utils.data import DataLoader
from src.model import LiteralEmbeddingsExt, LiteralEmbeddingsCliffordExt
from src.clifford import Keci
from dicee.static_funcs import  intialize_model

def get_callbacks(args):
    # Callbacks setup
    callbacks = [EpochLevelProgressBar(), BatchProcessCallback()]
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
            relation_idxs=entity_dataset.relation_to_idx,
            form="EntityPrediction"
        )
        if args.log_validation and not args.train_all_triples:
            valid_dataset = KvsAll(
                train_set_idx=entity_dataset.valid_set,
                entity_idxs=entity_dataset.entity_to_idx,
                relation_idxs=entity_dataset.relation_to_idx,
                form="EntityPrediction",
        )
    elif args.scoring_technique == "OneVsAll":
        train_dataset = OneVsAll(
            train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx,
            relation_idxs=entity_dataset.relation_to_idx,
            form="EntityPrediction"
        )
        if args.log_validation and not args.train_all_triples:
            valid_dataset = OneVsAll(
                train_set_idx=entity_dataset.valid_set,
                entity_idxs=entity_dataset.entity_to_idx,
                relation_idxs=entity_dataset.relation_to_idx,
                form="EntityPrediction",
            )
    else:
        raise ValueError(f"Unknown scoring technique: {args.scoring_technique}. Supported techniques: 'KvsAll', 'OneVsAll'")
    
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_core,
        pin_memory=True,        # faster transfer to GPU
        prefetch_factor=2,      # workers prefetch batches
        persistent_workers=True # workers stay alive across epochs
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


def get_model(args):
    if args.model == "Keci":
        kge_model = Keci(args=vars(args))
    else:
        kge_model, _ = intialize_model(vars(args), 0)
    return kge_model