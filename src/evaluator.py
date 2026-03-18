from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class Evaluator:
    """Local evaluator using the same core logic as Literal_EA/evaluate.py."""

    def __init__(self, args):
        self.args = args
        self.report: Dict[str, Dict[str, float]] = {}
        self.er_vocab: Dict[Tuple[int, int], List[int]] = {}

    def eval(self, dataset, trained_model, form_of_labelling="EntityPrediction", during_training=False):
        del form_of_labelling
        eval_mode = self._resolve_eval_mode()
        if eval_mode is None:
            return None
        eval_start_time = time.perf_counter()

        self.er_vocab = self._resolve_er_vocab(dataset)
        batch_size = int(getattr(self.args, "batch_size", 1024))

        train_loader = self._build_loader(getattr(dataset, "train_set", None), batch_size) if "train" in eval_mode else None
        val_loader = self._build_loader(getattr(dataset, "valid_set", None), batch_size) if "val" in eval_mode else None
        test_loader = self._build_loader(getattr(dataset, "test_set", None), batch_size) if "test" in eval_mode else None

        raw_report = self.evaluate(
            model=trained_model,
            eval_mode=eval_mode,
            batch_size=batch_size,
            log=not during_training,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            val_dataloader=val_loader,
        )

        mapped_report: Dict[str, Dict[str, float]] = {}
        if "train" in raw_report and raw_report["train"] is not None:
            mapped_report["Train"] = raw_report["train"]
        if "val" in raw_report and raw_report["val"] is not None:
            mapped_report["Val"] = raw_report["val"]
        if "test" in raw_report and raw_report["test"] is not None:
            mapped_report["Test"] = raw_report["test"]
        if "__meta__" in raw_report:
            mapped_report["__meta__"] = raw_report["__meta__"]
        total_eval_runtime = time.perf_counter() - eval_start_time
        mapped_report.setdefault("__meta__", {})
        mapped_report["__meta__"]["evaluation_wall_time_seconds_total"] = float(total_eval_runtime)
        self.report = mapped_report

        if not during_training:
            print("Total Evaluation Wall Time (s)")
            print(round(total_eval_runtime, 4))

        if not during_training:
            full_storage_path = getattr(self.args, "full_storage_path", None)
            if full_storage_path:
                os.makedirs(full_storage_path, exist_ok=True)
                with open(os.path.join(full_storage_path, "eval_report.json"), "w") as f:
                    json.dump(self.report, f, indent=4)

        return dict(self.report)

    def evaluate(
        self,
        model,
        eval_mode="test",
        batch_size=1024,
        log=True,
        train_dataloader=None,
        test_dataloader=None,
        val_dataloader=None,
    ):
        train_scores, test_scores, val_scores = None, None, None
        split_times = {}

        if "train" in eval_mode and train_dataloader is not None:
            t0 = time.perf_counter()
            train_scores = self.evaluate_link_prediction_performance(model, train_dataloader, batch_size=batch_size)
            split_times["train"] = time.perf_counter() - t0

        if "test" in eval_mode and test_dataloader is not None:
            t0 = time.perf_counter()
            test_scores = self.evaluate_link_prediction_performance(model, test_dataloader, batch_size=batch_size)
            split_times["test"] = time.perf_counter() - t0

        if "val" in eval_mode and val_dataloader is not None:
            t0 = time.perf_counter()
            val_scores = self.evaluate_link_prediction_performance(model, val_dataloader, batch_size=batch_size)
            split_times["val"] = time.perf_counter() - t0

        if log:
            if train_scores is not None:
                print("Train Scores")
                print(train_scores)
            if test_scores is not None:
                print("Test Scores")
                print(test_scores)
            if val_scores is not None:
                print("Validation Scores")
                print(val_scores)
            if split_times:
                print("Evaluation Runtime (s)")
                print({k: round(v, 4) for k, v in split_times.items()})

        eval_report = {}
        if train_scores is not None:
            eval_report["train"] = train_scores
        if test_scores is not None:
            eval_report["test"] = test_scores
        if val_scores is not None:
            eval_report["val"] = val_scores
        if split_times:
            eval_report["__meta__"] = {
                "evaluation_runtime_seconds_total": float(sum(split_times.values())),
                "evaluation_runtime_seconds_by_split": {k: float(v) for k, v in split_times.items()},
            }
        return eval_report

    @torch.no_grad()
    def evaluate_link_prediction_performance(self, model, data_loader, batch_size=1024):
        del batch_size
        model.eval()
        er_vocab = self.er_vocab
        reciprocal_rank_sum = 0.0
        hits1_count = 0
        hits3_count = 0
        hits10_count = 0
        device = torch.device("cpu")
        if hasattr(model, "parameters"):
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        if isinstance(device, str):
            device = torch.device(device)
        num_triples = 0

        for batch in data_loader:
            e1_idx, r_idx, e2_idx = batch
            e1_idx, r_idx, e2_idx = e1_idx.to(device), r_idx.to(device), e2_idx.to(device)
            batch_size_actual = e1_idx.size(0)
            num_triples += batch_size_actual

            if hasattr(model, "forward_k_vs_all"):
                hr = torch.stack((e1_idx, r_idx), dim=1)
                predictions = model.forward_k_vs_all(hr)
            else:
                try:
                    predictions = model(e1_idx, r_idx)
                except TypeError:
                    hr = torch.stack((e1_idx, r_idx), dim=1)
                    predictions = model(hr)

            # Keep target scores before filtering, then restore targets.
            target_scores = predictions.gather(1, e2_idx.view(-1, 1)).squeeze(1)

            # Apply filtered setting in a batched assignment.
            row_ids = []
            col_ids = []
            for j in range(batch_size_actual):
                head_idx = int(e1_idx[j].item())
                rel_idx = int(r_idx[j].item())
                tail_idx = int(e2_idx[j].item())
                filt = er_vocab.get((head_idx, rel_idx), [])
                if not filt:
                    continue
                for idx in filt:
                    idx = int(idx)
                    if idx != tail_idx:
                        row_ids.append(j)
                        col_ids.append(idx)
            if row_ids:
                rows = torch.as_tensor(row_ids, dtype=torch.long, device=predictions.device)
                cols = torch.as_tensor(col_ids, dtype=torch.long, device=predictions.device)
                predictions[rows, cols] = float("-inf")
            predictions.scatter_(1, e2_idx.view(-1, 1), target_scores.view(-1, 1))

            # Vectorized filtered ranks: rank = 1 + #entities with strictly higher score.
            ranks = (predictions > target_scores.unsqueeze(1)).sum(dim=1) + 1
            reciprocal_rank_sum += torch.reciprocal(ranks.to(torch.float32)).sum().item()
            hits1_count += (ranks <= 1).sum().item()
            hits3_count += (ranks <= 3).sum().item()
            hits10_count += (ranks <= 10).sum().item()

        hit_1 = (hits1_count / num_triples) if num_triples > 0 else 0.0
        hit_3 = (hits3_count / num_triples) if num_triples > 0 else 0.0
        hit_10 = (hits10_count / num_triples) if num_triples > 0 else 0.0
        mean_reciprocal_rank = (reciprocal_rank_sum / num_triples) if num_triples > 0 else 0.0
        return {"H@1": hit_1, "H@3": hit_3, "H@10": hit_10, "MRR": mean_reciprocal_rank}

    def _resolve_eval_mode(self) -> Optional[str]:
        eval_model = getattr(self.args, "eval_model", None)
        if eval_model in (None, "None"):
            return None
        if isinstance(eval_model, bool):
            eval_model = "train_val_test"
            self.args.eval_model = eval_model
        if "constraint" in eval_model:
            return eval_model.replace("_constraint", "")
        return eval_model

    @staticmethod
    def _build_loader(triples, batch_size: int) -> Optional[DataLoader]:
        if triples is None:
            return None
        triples_arr = np.asarray(triples)
        if triples_arr.size == 0:
            return None
        if triples_arr.ndim != 2 or triples_arr.shape[1] < 3:
            raise ValueError("Expected indexed triples with shape [N,3] (or wider).")
        heads = torch.as_tensor(triples_arr[:, 0], dtype=torch.long)
        rels = torch.as_tensor(triples_arr[:, 1], dtype=torch.long)
        tails = torch.as_tensor(triples_arr[:, 2], dtype=torch.long)
        return DataLoader(TensorDataset(heads, rels, tails), batch_size=batch_size, shuffle=False)

    @staticmethod
    def _resolve_er_vocab(dataset) -> Dict[Tuple[int, int], List[int]]:
        er_vocab = getattr(dataset, "er_vocab", None)
        if er_vocab is not None and not isinstance(er_vocab, dict) and hasattr(er_vocab, "result"):
            er_vocab = er_vocab.result()
        if isinstance(er_vocab, dict):
            return {(int(h), int(r)): [int(t) for t in tails] for (h, r), tails in er_vocab.items()}

        vocab: Dict[Tuple[int, int], set] = {}
        for attr in ("train_set", "valid_set", "test_set"):
            triples = getattr(dataset, attr, None)
            if triples is None:
                continue
            arr = np.asarray(triples)
            if arr.size == 0:
                continue
            for h, r, t in arr[:, :3]:
                key = (int(h), int(r))
                if key not in vocab:
                    vocab[key] = set()
                vocab[key].add(int(t))
        return {k: sorted(v) for k, v in vocab.items()}
