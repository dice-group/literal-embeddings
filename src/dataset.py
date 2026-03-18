import numpy as np
import torch
from dicee.static_preprocess_funcs import mapping_from_first_two_cols_to_third


class OnevsAllDataset(torch.utils.data.Dataset):
    """Dataset for the 1vsAll training strategy."""

    def __init__(self, train_set_idx: np.ndarray, entity_idxs, collate_fn=None):
        super().__init__()
        assert isinstance(train_set_idx, (np.memmap, np.ndarray))
        assert len(train_set_idx) > 0
        self.train_data = train_set_idx
        self.target_dim = len(entity_idxs)
        self.collate_fn = collate_fn
        self.labels = torch.zeros(len(self.train_data), self.target_dim)
        self.labels[torch.arange(len(self.train_data)), self.train_data[:, 2]] = 1

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        triple = torch.from_numpy(self.train_data[idx].copy()).long()
        return triple[:2], self.labels[idx], triple[2]


class KvsAll(torch.utils.data.Dataset):
    """Dataset for the KvsAll training strategy."""

    def __init__(self, train_set_idx: np.ndarray, entity_idxs, store=None, collate_fn=None):
        super().__init__()
        assert isinstance(train_set_idx, (np.memmap, np.ndarray))
        assert len(train_set_idx) > 0
        self.collate_fn = collate_fn
        self.target_dim = len(entity_idxs)

        if store is None:
            store = mapping_from_first_two_cols_to_third(train_set_idx)
        else:
            raise ValueError("Custom stores are not supported in the simplified KGE path.")
        assert len(store) > 0

        self.train_data = torch.LongTensor(list(store.keys()))
        if sum(len(i) for i in store.values()) == len(store):
            self.train_target = np.array(list(store.values()))
            try:
                assert isinstance(self.train_target[0], np.ndarray)
            except (IndexError, AssertionError):
                print(self.train_target)
                raise
        else:
            self.train_target = list(store.values())
            assert isinstance(self.train_target[0], list)

    def __len__(self):
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_target[idx]] = 1.0
        return self.train_data[idx], y_vec, self.train_target[idx]
