import torch
import pandas as pd
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

class LiteralDataset(Dataset):
    """Dataset for loading and processing literal data for training Literal Embedding model.
    This dataset handles the loading, normalization, and preparation of triples
    for training a literal embedding model.

    Extends torch.utils.data.Dataset for supporting PyTorch dataloaders.

    Attributes:
        train_file_path (str): Path to the training data file.
        normalization (str): Type of normalization to apply ('z-norm', 'min-max', or None).
        normalization_params (dict): Parameters used for normalization.
        entity_to_idx (dict): Mapping of entities to their indices.
        num_entities (int): Total number of entities.
        data_property_to_idx (dict): Mapping of data properties to their indices.
        num_data_properties (int): Total number of data properties.
    """

    def __init__(
        self,
        dataset_dir: str,
        ent_idx: dict = None,
        normalization_type: str = "z-norm",
    ):
        self.train_file_path = f"{dataset_dir}/literals/numerical_literals.txt"
        self.normalization_type = normalization_type
        self.normalization_params = {}
        self.entity_to_idx = ent_idx
        self.num_entities = len(self.entity_to_idx)

        if self.entity_to_idx is None:
            raise ValueError(
                "entity_to_idx must be provided to initialize LiteralDataset."
            )

        self._load_data()

    def _load_data(self):
        train_df = pd.read_csv(
            self.train_file_path, sep="\t", header=None, names=["head", "attribute", "value"]
        )
        train_df = train_df[train_df["head"].isin(self.entity_to_idx)]
        assert not train_df.empty, "Filtered train_df is empty — no entities match entity_to_idx."

        self.data_property_to_idx = {
            rel: idx
            for idx, rel in enumerate(sorted(train_df["attribute"].unique()))
        }
        self.num_data_properties = len(self.data_property_to_idx)

        train_df["head_idx"] = train_df["head"].map(self.entity_to_idx)
        train_df["attr_idx"] = train_df["attribute"].map(self.data_property_to_idx)
        train_df = self._apply_normalization(train_df)

        self.triples = torch.tensor(
            train_df[["head_idx", "attr_idx"]].values, dtype=torch.long
        )
        self.values = torch.tensor(train_df["value"].values, dtype=torch.float32)
        self.values_norm = torch.tensor(
            train_df["value_norm"].values, dtype=torch.float32
        )

    def _apply_normalization(self, df):
        """Applies normalization to the tail values based on the specified type."""
        if self.normalization_type == "z-norm":
            stats = df.groupby("attribute")["value"].agg(["mean", "std"])
            self.normalization_params = stats.to_dict(orient="index")
            df["value_norm"] = df.groupby("attribute")["value"].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            self.normalization_params["type"] = "z-norm"

        elif self.normalization_type == "min-max":
            stats = df.groupby("attribute")["value"].agg(["min", "max"])
            self.normalization_params = stats.to_dict(orient="index")
            df["value_norm"] = df.groupby("attribute")["value"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            self.normalization_params["type"] = "min-max"

        else:
            print(" No normalization applied.")
            df["value_norm"] = df["value"]
            if self.normalization_type is None:
                self.normalization_params = {}
                self.normalization_params["type"] = None


        return df

    def __getitem__(self, index):
        return self.triples[index], self.values_norm[index]

    def get_batch(self, entity_indices):
        """Get a batch of data for specific entity indices."""
        # Combine data and labels into one tensor of shape [n, d+1]
        combined = torch.cat([self.triples, self.values_norm.unsqueeze(1)], dim=1)
        src_device = entity_indices.device
        entity_indices = entity_indices.to('cpu')
        # Filter by whether first column matches values in entity_indices
        mask = torch.isin(combined[:, 0], entity_indices)
        filtered = combined[mask]

        # Select all triples for each entity (no random sampling)
        selected_triples = filtered

        # Extract entity indices, relation indices, and labels
        ent_ids = selected_triples[:, 0].long().to(src_device)
        rel_ids = selected_triples[:, 1].long().to(src_device)
        labels = selected_triples[:, 2].float().to(src_device)

        return ent_ids, rel_ids, labels

    def __len__(self):
        return len(self.triples)