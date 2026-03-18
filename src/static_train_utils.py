import torch
from dicee.static_funcs import intialize_model
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import (
    StochasticWeightAveraging as SWA,
)
from torch.utils.data import DataLoader

from src.callbacks import ASWA, EpochLevelProgressBar, PeriodicEvalCallback
from src.dataset import KvsAll, OnevsAllDataset


def get_callbacks(args):
    callbacks = [EpochLevelProgressBar()]
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="ent_loss_val",
                patience=args.early_stopping_patience,
                mode="min",
            )
        )
    if args.swa:
        callbacks.append(SWA(swa_epoch_start=1, swa_lrs=0.05))
    elif args.adaptive_swa:
        callbacks.append(ASWA(num_epochs=args.num_epochs, path=args.full_storage_path))
    else:
        print("No Stochastic Weight Averaging (SWA) or Adaptive SWA (ASWA) used.")

    if args.eval_every_n_epochs > 0 or args.eval_at_epochs is not None:
        callbacks.append(
            PeriodicEvalCallback(
                experiment_path=args.full_storage_path,
                max_epochs=args.num_epochs,
                eval_every_n_epoch=args.eval_every_n_epochs,
                eval_at_epochs=args.eval_at_epochs,
                save_model_every_n_epoch=args.save_every_n_epochs,
                n_epochs_eval_model=args.n_epochs_eval_model,
            )
        )
    return callbacks


def kvsall_collate_fn(batch):
    xs, y_vecs, targets = zip(*batch)
    xs = torch.stack(xs)
    y_vecs = torch.stack(y_vecs)
    targets = [torch.as_tensor(t, dtype=torch.long) for t in targets]
    targets = torch.cat(targets, dim=0)
    return xs, y_vecs, targets


def _build_dataloader(dataset, batch_size, shuffle, num_workers, collate_fn=None):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs.update(
            {
                "pin_memory": True,
                "prefetch_factor": 2,
                "persistent_workers": True,
            }
        )
    return DataLoader(**loader_kwargs)


def get_dataloaders(args, entity_dataset):
    if args.scoring_technique == "KvsAll":
        train_dataset = KvsAll(
            train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx,
            collate_fn=kvsall_collate_fn,
        )
        valid_dataset = None
        if args.log_validation and not args.train_all_triples:
            valid_dataset = KvsAll(
                train_set_idx=entity_dataset.valid_set,
                entity_idxs=entity_dataset.entity_to_idx,
            )
    elif args.scoring_technique == "1vsAll":
        train_dataset = OnevsAllDataset(
            train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx,
            collate_fn=None,
        )
        valid_dataset = None
        if args.log_validation and not args.train_all_triples:
            valid_dataset = OnevsAllDataset(
                train_set_idx=entity_dataset.valid_set,
                entity_idxs=entity_dataset.entity_to_idx,
            )
    else:
        raise ValueError(
            f"Unknown scoring technique: {args.scoring_technique}. Supported "
            "techniques: 'KvsAll', '1vsAll'"
        )

    train_dataloader = _build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_core,
        collate_fn=train_dataset.collate_fn,
    )
    valid_dataloader = None
    if valid_dataset is not None:
        valid_dataloader = _build_dataloader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(args.num_core, 1),
            collate_fn=valid_dataset.collate_fn,
        )
    return train_dataloader, valid_dataloader


def get_model(args, entity_dataset=None):
    del entity_dataset
    kge_model, _ = intialize_model(vars(args), 0)
    return kge_model
