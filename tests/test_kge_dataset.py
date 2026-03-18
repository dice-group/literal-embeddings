import numpy as np
import torch

from src.dataset import KvsAll, OnevsAllDataset


def test_onevsall_dataset_encodes_tail_labels():
    train_set = np.array([[0, 0, 1], [2, 1, 3]], dtype=np.int64)
    dataset = OnevsAllDataset(train_set_idx=train_set, entity_idxs={str(i): i for i in range(4)})

    triple, labels, tail = dataset[0]

    assert torch.equal(triple, torch.tensor([0, 0]))
    assert tail == 1
    assert labels.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_kvsall_dataset_groups_targets_per_entity_relation_pair():
    train_set = np.array([[0, 0, 1], [0, 0, 2], [1, 1, 3]], dtype=np.int64)
    dataset = KvsAll(train_set_idx=train_set, entity_idxs={str(i): i for i in range(4)})

    assert len(dataset) == 2

    first_pair, first_labels, first_targets = dataset[0]
    assert first_pair.tolist() in ([0, 0], [1, 1])
    if first_pair.tolist() == [0, 0]:
        assert first_labels.tolist() == [0.0, 1.0, 1.0, 0.0]
        assert list(first_targets) == [1, 2]
