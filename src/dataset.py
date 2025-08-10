import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class LiteralDataset(Dataset):
    def __init__(self, dataset_dir, ent_idx, normalization="z-norm", sampling_ratio=None, selected_attributes=None, 
                 label_perturbation=None, perturbation_ratio=0.1, perturbation_noise_std=0.1, random_seed=42):
        """
        Initialize LiteralDataset with optional label perturbation.
        
        Args:
            dataset_dir: Path to dataset directory
            ent_idx: Entity index mapping
            normalization: Normalization type ("z-norm", "min-max", etc.)
            sampling_ratio: Fraction of training data to use
            selected_attributes: List of specific attributes to use.
        """
        self.dataset_dir = os.path.join(dataset_dir, "literals")
        self.normalization_type = normalization
        self.normalization_params = {}
        self.sampling_ratio = sampling_ratio 
        self.selected_attributes = selected_attributes  # List of attribute names to filter
        
        # Label perturbation parameters
        self.label_perturbation = label_perturbation
        self.perturbation_ratio = perturbation_ratio
        self.perturbation_noise_std = perturbation_noise_std
        self.random_seed = random_seed
        self.perturbed_indices = set()  # Track which labels were perturbed

        self.file_paths = {
            s: os.path.join(self.dataset_dir, f"{s}.txt")
            for s in ("train", "val", "test")
        }

        self.entity_to_idx = (
            ent_idx
            if isinstance(ent_idx, dict)
            else {v: i for i, v in ent_idx["entity"].items()}
        )
        self.num_entities = len(self.entity_to_idx)

        self._load_data()

    def _load_data(self):
        # Load mapping from train
        train_path = self.file_paths["train"]
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Data file not found at {train_path}")
        train_df = pd.read_csv(
            train_path, sep="\t", header=None, names=["head", "relation", "tail"]
        )
        train_df = train_df[train_df["head"].isin(self.entity_to_idx)]
        
        # Filter by selected attributes if provided
        if self.selected_attributes is not None:
            original_count = len(train_df)
            train_df = train_df[train_df["relation"].isin(self.selected_attributes)]
            filtered_count = len(train_df)
            print(f"Filtered dataset from {original_count} to {filtered_count} triples using {len(self.selected_attributes)} selected attributes.")
            print(f"Selected attributes: {self.selected_attributes}")
            
            if filtered_count == 0:
                raise ValueError(f"No triples found for selected attributes: {self.selected_attributes}")
        
        self.data_property_to_idx = {
            rel: idx for idx, rel in enumerate(train_df["relation"].unique())
        }

        # reduce the train set for ablations using sampling ratio
        # keeps the sampling_ratio * 100 % of full training set in the train_df

        if self.sampling_ratio is not None:
            if not (0 < self.sampling_ratio <= 1):
                raise ValueError("Fraction must be between 0 and 1.")

            train_df = (
                train_df.groupby("relation", group_keys=False)
                .sample(frac=self.sampling_ratio, random_state=42)
    )
            print(f"Training Literal Embedding model with {self.sampling_ratio*100:.1f}% of the train set.")

        self.num_data_properties = len(self.data_property_to_idx)

        train_df["head_idx"] = train_df["head"].map(self.entity_to_idx)
        train_df["rel_idx"] = train_df["relation"].map(self.data_property_to_idx)
        
        # Apply normalization FIRST, then perturbation
        train_df = self._apply_normalization(train_df)

        self.triples = torch.tensor(
            train_df[["head_idx", "rel_idx"]].values, dtype=torch.long
        )
        self.tails = torch.tensor(train_df["tail"].values, dtype=torch.float32)
        self.tails_norm = torch.tensor(
            train_df["tail_norm"].values, dtype=torch.float32
        )

    def _apply_normalization(self, df):
        """Applies normalization to the tail values based on the specified type."""
        if self.normalization_type == "z-norm":
            stats = df.groupby("relation")["tail"].agg(["mean", "std"])
            self.normalization_params = stats.to_dict(orient="index")
            df["tail_norm"] = df.groupby("relation")["tail"].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            self.normalization_params["type"] = "z-norm"

        elif self.normalization_type == "min-max":
            stats = df.groupby("relation")["tail"].agg(["min", "max"])
            self.normalization_params = stats.to_dict(orient="index")
            df["tail_norm"] = df.groupby("relation")["tail"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            self.normalization_params["type"] = "min-max"

        else:
            print(" No normalization applied.")
            df["tail_norm"] = df["tail"]
            if self.normalization_type is None:
                self.normalization_params = {}
                self.normalization_params["type"] = None


        return df

    def __getitem__(self, index):
        return self.triples[index], self.tails_norm[index]

    def __len__(self):
        return len(self.triples)

    def get_df(self, split="test", norm=False):
        file_path = self.file_paths.get(split)
        if file_path is None or not os.path.isfile(file_path):
            raise ValueError(
                f"Invalid data split or path does not lead to a file: {file_path}"
            )

        df = pd.read_csv(
            file_path, sep="\t", header=None, names=["head", "relation", "tail"]
        )
        df = df[df["head"].isin(self.entity_to_idx)]
        
        # Filter by selected attributes if provided
        if self.selected_attributes is not None:
            original_count = len(df)
            df = df[df["relation"].isin(self.selected_attributes)]
            print(f"Filtered {split} set from {original_count} to {len(df)} triples using selected attributes.")

        df["head_idx"] = df["head"].map(self.entity_to_idx)
        df["rel_idx"] = df["relation"].map(self.data_property_to_idx)
        
        # Filter out relations not in training set (important when using selected_attributes)
        df = df.dropna(subset=['rel_idx'])

        return df

    def build_entity_index(self):
        # Build a mapping from entity id to row indices for fast lookup
        
        self.entity_to_rows = defaultdict(list)
        for idx, ent_id in enumerate(self.triples[:, 0].tolist()):
            self.entity_to_rows[ent_id].append(idx)

    def get_batch(self, entity_indices, multi_regression=False):
        # Ensure entity_to_rows is built
        if not hasattr(self, "entity_to_rows"):
            self.build_entity_index()

        # Flatten all row indices for requested entities
        entity_indices = entity_indices.tolist() if isinstance(entity_indices, torch.Tensor) else entity_indices
        row_indices = []
        for ent in entity_indices:
            row_indices.extend(self.entity_to_rows.get(ent, []))
        if not row_indices:
            # No matching entities
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32)

        selected_triples = self.triples[row_indices]
        selected_tails_norm = self.tails_norm[row_indices]

        ent_ids = selected_triples[:, 0].long()
        rel_ids = selected_triples[:, 1].long()
        labels = selected_tails_norm.float()

        if multi_regression:
            y_true = torch.full(
                (len(ent_ids), self.num_data_properties), -9.0, dtype=torch.float32
            )
            y_true[torch.arange(len(ent_ids)), rel_ids] = labels
        else:
            y_true = labels

        return ent_ids, rel_ids, y_true


    def get_ea_encoding(self):
        ea = torch.zeros(
            self.num_entities, self.num_data_properties, dtype=torch.float32
        )
        e = self.triples[:, 0]
        a = self.triples[:, 1]
        ea[e, a] = 1.0
        return ea
    
    def get_available_attributes(self):
        """Get list of all available attributes in the training data"""
        train_path = self.file_paths["train"]
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Data file not found at {train_path}")
        
        train_df = pd.read_csv(
            train_path, sep="\t", header=None, names=["head", "relation", "tail"]
        )
        return sorted(train_df["relation"].unique().tolist())
    
    def get_attribute_stats(self):
        """Get statistics about attributes in the dataset"""
        train_path = self.file_paths["train"]
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Data file not found at {train_path}")
        
        train_df = pd.read_csv(
            train_path, sep="\t", header=None, names=["head", "relation", "tail"]
        )
        train_df = train_df[train_df["head"].isin(self.entity_to_idx)]
        
        # If filtering by selected attributes, show both original and filtered stats
        stats = {}
        if self.selected_attributes is not None:
            # Original stats
            original_stats = train_df["relation"].value_counts()
            stats["original"] = original_stats
            
            # Filtered stats
            filtered_df = train_df[train_df["relation"].isin(self.selected_attributes)]
            filtered_stats = filtered_df["relation"].value_counts()
            stats["filtered"] = filtered_stats
        else:
            stats["all"] = train_df["relation"].value_counts()
        
        return stats
    
    @staticmethod
    def denormalize( preds_norm, attributes, normalization_params) -> np.ndarray:
        """Denormalizes the predictions based on the normalization type.

        Args:
        preds_norm (np.ndarray): Normalized predictions to be denormalized.
        attributes (list): List of attributes corresponding to the predictions.
        normalization_params (dict): Dictionary containing normalization parameters for each attribute.

        Returns:
            np.ndarray: Denormalized predictions.

        """
        if normalization_params["type"] == "z-norm":
            # Extract means and stds only if z-norm is used
            means = np.array([normalization_params[i]["mean"] for i in attributes])
            stds = np.array([normalization_params[i]["std"] for i in attributes])
            return preds_norm * stds + means

        elif normalization_params["type"] == "min-max":
            # Extract mins and maxs only if min-max is used
            mins = np.array([normalization_params[i]["min"] for i in attributes])
            maxs = np.array([normalization_params[i]["max"] for i in attributes])
            return preds_norm * (maxs - mins) + mins

        elif normalization_params["type"] is None:
            return  preds_norm  # No normalization applied, return as is

        else:
            raise ValueError(
                f"Unsupported normalization type: {normalization_params['type']}. "
                f"Supported types: 'z-norm', 'min-max', or None."
            )
