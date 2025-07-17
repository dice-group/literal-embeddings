import torch
from torch.utils.data import Dataset

class KGDataset(Dataset):
    """PyTorch Dataset for Knowledge Graph triples with consistent entity/relation mappings"""
    
    def __init__(self, data_dir="data/FB15K-237/", split="train", reverse=False):
        """
        Args:
            data_dir: directory containing train.txt, valid.txt, test.txt
            split: which split to use ("train", "valid", "test")
            reverse: whether to add reverse triples
        """
        # Load all splits to create consistent mappings
        self.splits = {}
        for split_name in ['train', 'valid', 'test']:
            self.splits[split_name] = self._load_raw(data_dir, split_name, reverse)
        
        # Create global entity and relation vocabularies
        self.all_triples = self.splits['train'] + self.splits['valid'] + self.splits['test']
        self.entities = sorted(set(h for h,_,t in self.all_triples) | set(t for h,_,t in self.all_triples))
        self.relations = sorted(set(r for _,r,_ in self.all_triples))
        
        # Create index mappings
        self.entity2idx = {e: i for i, e in enumerate(self.entities)}
        self.relation2idx = {r: i for i, r in enumerate(self.relations)}

        # Select triples for this specific split
        self.triples = self.splits[split]

    def _load_raw(self, data_dir, split, reverse):
        """Load raw triples from file"""
        triples = []
        with open(f"{data_dir}/{split}.txt", "r") as f:
            for line in f:
                h, r, t = line.strip().split()
                triples.append((h,r,t))
        if reverse:
            triples += [(t, r + "_reverse", h) for (h, r, t) in triples]
        return triples

    def __len__(self):
        """Return the number of triples in the dataset"""
        return len(self.triples)

    def __getitem__(self, idx):
        """Return triple as (head_idx, relation_idx, tail_idx)"""
        h, r, t = self.triples[idx]
        return (self.entity2idx[h], self.relation2idx[r], self.entity2idx[t])