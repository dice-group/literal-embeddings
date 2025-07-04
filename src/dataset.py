import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


class LiteralDataset(Dataset):
    def __init__(self, dataset_dir, ent_idx, normalization="z-norm", sampling_ratio = None):
        self.dataset_dir = os.path.join(dataset_dir, "literals")
        self.normalization_type = normalization
        self.normalization_params = {}
        self.sampling_ratio = sampling_ratio 

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
                .apply(lambda x: x.sample(frac=self.sampling_ratio, random_state=42))
                .reset_index(drop=True)
            )
            print(f"Training Literal Embedding model with {self.sampling_ratio*100:.1f}% of the train set.")


        self.num_data_properties = len(self.data_property_to_idx)

        train_df["head_idx"] = train_df["head"].map(self.entity_to_idx)
        train_df["rel_idx"] = train_df["relation"].map(self.data_property_to_idx)
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

        df["head_idx"] = df["head"].map(self.entity_to_idx)
        df["rel_idx"] = df["relation"].map(self.data_property_to_idx)

        if norm:
            df = self._apply_normalization(df)

        return df

    def get_batch(self, entity_indices, multi_regression=False, random_seed=None):
        # Combine data and labels into one tensor of shape [n, d+1]
        combined = torch.cat([self.triples, self.tails_norm.unsqueeze(1)], dim=1)

        # Filter by whether first column matches values in entity_indices
        mask = torch.isin(combined[:, 0], entity_indices)
        filtered = combined[mask]

        if random_seed is not None:
            # Use the provided random seed to shuffle the filtered triples
            torch.manual_seed(random_seed)
            shuffled = filtered[torch.randperm(filtered.size(0))]

            # Group by first column (entity indices) and select the first row of each group
            values = shuffled[:, 0]
            unique_vals, inverse_indices = torch.unique(values, return_inverse=True)
            first_indices = torch.stack(
                [
                    (inverse_indices == i).nonzero(as_tuple=True)[0][0]
                    for i in range(len(unique_vals))
                ]
            )
            selected_triples = shuffled[first_indices]
        else:
            # Select all triples for each entity
            selected_triples = filtered

        # Extract entity indices, relation indices, and labels
        ent_ids = selected_triples[:, 0].long()
        rel_ids = selected_triples[:, 1].long()
        labels = selected_triples[:, 2].float()

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
                "Unsupported normalization type. Use 'z-norm', 'min-max', or None."
            )
