from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.evaluator import Evaluator as LocalEvaluator


try:
    from dicee.evaluation.evaluator import Evaluator as DiceEvaluator
except ImportError:  # pragma: no cover - environment-dependent
    DiceEvaluator = None


class _DummyDataset:
    def __init__(self):
        self.num_entities = 12
        self.num_relations = 4

        self.train_set = np.array(
            [
                [0, 0, 1],
                [0, 0, 2],
                [1, 1, 3],
                [2, 2, 4],
            ],
            dtype=np.int64,
        )
        self.valid_set = np.array(
            [
                [3, 1, 5],
                [4, 2, 6],
            ],
            dtype=np.int64,
        )
        self.test_set = np.array(
            [
                [5, 1, 7],
                [6, 2, 8],
                [7, 3, 9],
            ],
            dtype=np.int64,
        )

        self.er_vocab = self._build_er_vocab()
        # Not used in entity-prediction path, but required by dice evaluator.
        self.re_vocab = {}
        self.ee_vocab = {}
        self.func_triple_to_bpe_representation = None

    def _build_er_vocab(self):
        vocab = {}
        for triples in (self.train_set, self.valid_set, self.test_set):
            for h, r, t in triples:
                vocab.setdefault((int(h), int(r)), set()).add(int(t))
        return {k: sorted(v) for k, v in vocab.items()}


class _DummyKvsAllModel(torch.nn.Module):
    def __init__(self, num_entities: int):
        super().__init__()
        self.num_entities = num_entities
        self.name = "DummyKvsAll"

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # Deterministic, near tie-free row-wise scores.
        # x is [batch, 2] = (head, relation).
        x = x.to(torch.float32)
        h = x[:, 0:1]
        r = x[:, 1:2]
        entity_ids = torch.arange(self.num_entities, device=x.device, dtype=torch.float32).view(1, -1)
        scores = entity_ids + (0.017 * h) + (0.013 * r) + (h * r * 1e-4)
        return scores


@pytest.mark.skipif(DiceEvaluator is None, reason="dicee is not installed in this test environment")
def test_local_evaluator_matches_dicee_on_entity_prediction(tmp_path):
    dataset = _DummyDataset()
    model = _DummyKvsAllModel(num_entities=dataset.num_entities)

    base_args = dict(
        eval_model="train_val_test",
        num_folds_for_cv=0,
        scoring_technique="KvsAll",
        byte_pair_encoding=False,
        batch_size=4,
        model="Keci",
    )

    dice_path = tmp_path / "dice"
    local_path = tmp_path / "local"
    dice_path.mkdir(parents=True, exist_ok=True)
    local_path.mkdir(parents=True, exist_ok=True)

    dice_args = SimpleNamespace(**base_args, full_storage_path=str(dice_path))
    local_args = SimpleNamespace(**base_args, full_storage_path=str(local_path))

    dice_report = DiceEvaluator(args=dice_args).eval(
        dataset=dataset,
        trained_model=model,
        form_of_labelling="EntityPrediction",
        during_training=True,
    )
    local_report = LocalEvaluator(args=local_args).eval(
        dataset=dataset,
        trained_model=model,
        form_of_labelling="EntityPrediction",
        during_training=True,
    )

    for split in ("Train", "Val", "Test"):
        for metric in ("MRR", "H@1", "H@3", "H@10"):
            assert local_report[split][metric] == pytest.approx(dice_report[split][metric], abs=1e-6)
