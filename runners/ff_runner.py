import os
import json
import torch

from tqdm import tqdm
from dicee.evaluator import Evaluator
from dicee.knowledge_graph import KG
from torch.utils.data import DataLoader
from dicee.static_funcs import  read_or_load_kg, store, save_checkpoint_model
from src.static_funcs import  save_kge_experiments, get_full_storage_path
from src.static_train_utils import   get_ff_models
from src.dataset import FFKvsAllDataset, NegSampleDataset, TriplePredictionDataset

def train_kge_ff(args):
    """Train a KGE model with ff-update."""
    # Set up experiment storage path
    args.learning_rate = args.lr

    ff_supported_scoring = {"NegSample", "KvsAll"}
    assert args.scoring_technique in ff_supported_scoring, (
        "Forward_forward supports scoring_technique in {'NegSample','KvsAll'}"
    )
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

    # train dataloader
    if args.scoring_technique == "NegSample":
        train_dataset = TriplePredictionDataset(
            neg_sample_ratio=args.neg_ratio,
            num_entities=args.num_entities,
            num_relations=args.num_relations,
            train_set=entity_dataset.train_set,
            seed=args.random_seed,
            filtered_negatives=getattr(args, "ff_filtered_negatives", False),
            hard_negative_ratio=getattr(args, "ff_hard_negative_ratio", 0.0),
            max_filter_retries=getattr(args, "ff_max_filter_retries", 10),
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_core,
            collate_fn=train_dataset.collate_fn,
        )
    else:
        train_dataset = FFKvsAllDataset(
            train_set_idx=entity_dataset.train_set,
            entity_idxs=entity_dataset.entity_to_idx,
            seed=args.random_seed,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_core,
            collate_fn=train_dataset.collate_fn,
        )
    supported_models = ["DistMult", "CLNN", "FFKGE"]
    assert args.model in supported_models , f"{args.model} not supported for forward-forward training"

    kge_model = get_ff_models(args)
    kge_model.to(args.device)
    evaluator = Evaluator(args=args)

    periodic_reports = {}
    eval_epochs = set()
    if getattr(args, "eval_every_n_epochs", 0) > 0:
        eval_epochs.update(range(args.eval_every_n_epochs, args.num_epochs + 1, args.eval_every_n_epochs))
    if getattr(args, "eval_at_epochs", None):
        eval_epochs.update(args.eval_at_epochs)
    periodic_eval_mode = getattr(args, "n_epochs_eval_model", args.eval_model)
    models_n_epochs_path = None
    if getattr(args, "save_every_n_epochs", False):
        models_n_epochs_path = os.path.join(args.full_storage_path, "models_n_epochs")
        os.makedirs(models_n_epochs_path, exist_ok=True)

    optimizer = torch.optim.Adam(kge_model.parameters(), lr=args.lr)
    for epoch in (tqdm_bar := tqdm(range(args.num_epochs))):
        kge_model.train()
        epoch_loss = 0
        epoch_hid_pos = 0.0
        epoch_hid_neg = 0.0
        epoch_out_pos = 0.0
        epoch_out_neg = 0.0
        has_components = False
        for batch in train_dataloader:
            if args.scoring_technique == "NegSample":
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    x_pos, x_neg = batch
                    # Backward compatibility for old NegSampleDataset stacked output: (B,2,3)
                    if x_pos.ndim == 3 and x_pos.size(1) == 2:
                        stacked = x_pos.to(args.device)
                        x_pos = stacked[:, 0]
                        x_neg = stacked[:, 1]
                    else:
                        x_pos = x_pos.to(args.device)
                        x_neg = x_neg.to(args.device)
                else:
                    raise ValueError("Unexpected FF batch format. Expected (x_pos, x_neg).")
                batch_report = kge_model.ff_update(x_pos, x_neg, optimizer)
            else:
                if not (isinstance(batch, (tuple, list)) and len(batch) == 3):
                    raise ValueError("Unexpected KvsAll FF batch format. Expected (hr, y_vec, targets).")
                x_hr, y_vec, target_tails = batch
                x_hr = x_hr.to(args.device)
                y_vec = y_vec.to(args.device)
                if not hasattr(kge_model, "ff_update_kvsall"):
                    raise AttributeError(
                        f"{args.model} does not implement ff_update_kvsall required for FF+KvsAll."
                    )
                batch_report = kge_model.ff_update_kvsall(
                    x_hr=x_hr,
                    y_vec=y_vec,
                    target_tails=target_tails,
                    optimizer=optimizer,
                    num_entities=args.num_entities,
                    num_negatives=getattr(args, "ff_kvsall_num_negatives", 32),
                    use_all_negatives=getattr(args, "ff_kvsall_use_all_negatives", False),
                )
            epoch_loss += batch_report["loss"]
            if "hid_pos_loss" in batch_report:
                has_components = True
                epoch_hid_pos += batch_report["hid_pos_loss"]
                epoch_hid_neg += batch_report["hid_neg_loss"]
                epoch_out_pos += batch_report["out_pos_loss"]
                epoch_out_neg += batch_report["out_neg_loss"]
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if has_components:
            avg_hid_pos = epoch_hid_pos / len(train_dataloader)
            avg_hid_neg = epoch_hid_neg / len(train_dataloader)
            avg_out_pos = epoch_out_pos / len(train_dataloader)
            avg_out_neg = epoch_out_neg / len(train_dataloader)
            tqdm_bar.set_postfix_str(
                " ff_loss={:.5f} hid(p/n)={:.4f}/{:.4f} out(p/n)={:.4f}/{:.4f}".format(
                    avg_epoch_loss, avg_hid_pos, avg_hid_neg, avg_out_pos, avg_out_neg
                )
            )
        else:
            tqdm_bar.set_postfix_str(f" ff_loss={avg_epoch_loss:.5f}")

        current_epoch = epoch + 1
        if current_epoch in eval_epochs:
            default_eval_model = evaluator.args.eval_model
            evaluator.args.eval_model = periodic_eval_mode

            model_device = next(kge_model.parameters()).device
            model_train_mode = kge_model.training
            kge_model.to("cpu")
            kge_model.eval()
            report = evaluator.eval(
                dataset=entity_dataset,
                trained_model=kge_model,
                form_of_labelling="EntityPrediction",
                during_training=True,
            )
            periodic_reports[f"epoch_{current_epoch}_eval"] = report
            kge_model.to(model_device)
            kge_model.train(mode=model_train_mode)
            evaluator.args.eval_model = default_eval_model

            if models_n_epochs_path is not None:
                save_path = os.path.join(models_n_epochs_path, f"model_at_epoch_{current_epoch}.pt")
                save_checkpoint_model(kge_model, path=save_path)

    kge_model.to("cpu")
    final_eval_report = evaluator.eval(
        dataset=entity_dataset,
        trained_model=kge_model,
        form_of_labelling="EntityPrediction",
    )
    print(f"Forward-forward  training for the Model {args.model} completed.")
    if periodic_reports:
        report_path = os.path.join(args.full_storage_path, "eval_report_n_epochs.json")
        with open(report_path, "w") as f:
            json.dump(periodic_reports, f, indent=4)
    save_kge_experiments(args=args, loss_log=None)
    return {"final_eval": final_eval_report, "periodic_eval": periodic_reports}
