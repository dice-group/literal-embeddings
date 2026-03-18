import os

from pytorch_lightning import Trainer
from torch.utils.data import random_split

from src.KGEntText.dataset import KGEntTextDataset
from src.KGEntText.eval import print_test_predictions, save_test_predictions
from src.KGEntText.text_pipeline import build_text_dataloader, load_text_tokenizer
from src.KGEntText.utils import (
    load_fb15k_entity_name_mapping,
    load_pretrained_kge_components,
)
from src.trainer_KGEntText import KGEntTextTrainer


def _split_text_dataset(dataset, seed, test_ratio=0.05):
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Need at least 2 text samples for train/test splitting.")

    test_size = max(1, int(dataset_size * test_ratio))
    train_size = dataset_size - test_size

    generator = None
    if seed is not None:
        import torch

        generator = torch.Generator().manual_seed(seed)

    return random_split(
        dataset,
        lengths=[train_size, test_size],
        generator=generator,
    )


def _resolve_num_workers(args):
    configured_workers = getattr(args, "num_core", 0)
    if configured_workers and configured_workers > 0:
        return configured_workers

    if hasattr(os, "sched_getaffinity"):
        available_cores = len(os.sched_getaffinity(0))
    else:
        available_cores = os.cpu_count() or 1
    return max(1, int(available_cores * 0.7))


def train_kgenttext_model(args):
    """Run minimal end-to-end KGEntText training."""
    pretrained_kge_path = args.pretrained_kge_path
    kge_components = load_pretrained_kge_components(
        pretrained_kge_path=pretrained_kge_path
    )
    ent_to_idx = kge_components["entity_to_idx"]
    entity_names = {}
    if "FB15k-237" in args.dataset_dir:
        try:
            entity_names = load_fb15k_entity_name_mapping()
        except Exception as exc:
            print(f"Could not load FB15k entity-name mapping, falling back to raw ids: {exc}")
    ent_desc_dataset = KGEntTextDataset(
        dataset_dir=args.dataset_dir,
        file_name=args.text_file_name,
        entity_to_idx=ent_to_idx,
        entity_names=entity_names,
    )
    if len(ent_desc_dataset) == 0:
        raise ValueError("KGEntText dataset is empty after filtering.")

    max_seq_len = getattr(args, "max_seq_len", 64)
    num_workers = _resolve_num_workers(args)
    tokenizer = load_text_tokenizer(getattr(args, "text_model_name", "google/gemma-3-1b-it"))
    train_dataset, test_dataset = _split_text_dataset(
        ent_desc_dataset,
        seed=getattr(args, "random_seed", None),
    )
    train_dataloader = build_text_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=max_seq_len,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dataloader = build_text_dataloader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=max_seq_len,
        shuffle=False,
        num_workers=min(num_workers, 1),
    )

    trainer_module = KGEntTextTrainer(
        args=args,
        kge_embeddings=kge_components["entity_embeddings"],
        tokenizer=tokenizer,
    )
    lightning_trainer = Trainer(
        max_epochs=args.num_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    lightning_trainer.fit(
        trainer_module,
        train_dataloaders=train_dataloader,
    )
    test_metrics = lightning_trainer.test(
        trainer_module,
        dataloaders=test_dataloader,
        verbose=False,
    )
    prediction_rows = print_test_predictions(
        model=trainer_module.model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        max_examples=10,
    )
    output_dir = args.full_storage_path or pretrained_kge_path
    predictions_path = os.path.join(output_dir, "kgenttext_test_predictions.jsonl")
    save_test_predictions(prediction_rows, predictions_path)

    print(
        "KGEntText components initialized:",
        {
            "num_text_samples": len(ent_desc_dataset),
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "num_entities": len(ent_to_idx),
            "num_workers": num_workers,
            "text_model_name": getattr(args, "text_model_name", "google/gemma-3-1b-it"),
            "vocab_size": tokenizer.vocab_size,
            "kge_embedding_dim": trainer_module.model.kge_dim,
            "model_hidden_dim": trainer_module.model.hidden_dim,
            "predictions_path": predictions_path,
            "test_metrics": test_metrics[0] if test_metrics else {},
        },
    )
    return {
        "dataset": ent_desc_dataset,
        "tokenizer": tokenizer,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "lightning_trainer": lightning_trainer,
        "trainer_module": trainer_module,
        "kge_components": kge_components,
        "predictions_path": predictions_path,
        "test_metrics": test_metrics,
    }
