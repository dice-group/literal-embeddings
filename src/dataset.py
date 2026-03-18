import os

import numpy as np
import torch
from dicee.static_preprocess_funcs import mapping_from_first_two_cols_to_third

class OnevsAllDataset(torch.utils.data.Dataset):
    """
       Dataset for the 1vsALL training strategy

       Parameters
       ----------
       train_set_idx
           Indexed triples for the training.
       entity_idxs
           mapping of entity indices.
       Returns
       -------
       torch.utils.data.Dataset
       """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs):
        super().__init__()
        assert isinstance(train_set_idx, np.memmap) or isinstance(train_set_idx, np.ndarray)
        assert len(train_set_idx) > 0
        self.train_data = train_set_idx
        self.target_dim = len(entity_idxs)
        self.collate_fn = None
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        y_vec = torch.zeros(self.target_dim)
        triple= torch.from_numpy(self.train_data[idx].copy()).long()
        y_vec[triple[2]] = 1
        return triple[:2], y_vec, triple[2]


class KvsAll(torch.utils.data.Dataset):
    """ Creates a dataset for KvsAll training by inheriting from torch.utils.data.Dataset.
    Let D denote a dataset for KvsAll training and be defined as D:= {(x,y)_i}_i ^N, where
    x: (h,r) is an unique tuple of an entity h \in E and a relation r \in R that has been seed in the input graph.
    y: denotes a multi-label vector \in [0,1]^{|E|} is a binary label. \forall y_i =1 s.t. (h r E_i) \in KG

    Parameters
    ----------
    train_set_idx : numpy.ndarray
        n by 3 array representing n triples

    entity_idxs : dictonary
        string representation of an entity to its integer id

    relation_idxs : dictonary
        string representation of a relation to its integer id

    Returns
    -------
    self : torch.utils.data.Dataset

    """

    def __init__(self, train_set_idx: np.ndarray, entity_idxs, store=None):
        super().__init__()
        assert len(train_set_idx) > 0
        assert isinstance(train_set_idx, np.memmap) or isinstance(train_set_idx, np.ndarray)
        self.train_data = None
        self.train_target = None
        self.collate_fn = None

        # (1) Create a dictionary of training data pints
        if store is None:
            store = dict()
            self.target_dim = len(entity_idxs)
            store = mapping_from_first_two_cols_to_third(train_set_idx)
           
        else:
            raise ValueError()
        assert len(store) > 0
        # Keys in store correspond to integer representation (index) of subject and predicate
        # Values correspond to a list of integer representations of entities.
        self.train_data = torch.LongTensor(list(store.keys()))

        if sum([len(i) for i in store.values()]) == len(store):
            # if each s,p pair contains at most 1 entity
            self.train_target = np.array(list(store.values()))
            try:
                assert isinstance(self.train_target[0], np.ndarray)
            except IndexError or AssertionError:
                print(self.train_target)
                # TODO: Add info
                exit(1)
        else:
            self.train_target = list(store.values())
            assert isinstance(self.train_target[0], list)
        del store

    def __len__(self):
        assert len(self.train_data) == len(self.train_target)
        return len(self.train_data)

    def __getitem__(self, idx):
        # 1. Initialize a vector of output.
        y_vec = torch.zeros(self.target_dim)
        y_vec[self.train_target[idx]] = 1.0
        return self.train_data[idx], y_vec, self.train_target[idx]
