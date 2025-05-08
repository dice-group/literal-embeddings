import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class LiteralDataset(Dataset):
    def __init__(self, dataset_dir, ent_idx, normalization="z-norm", sampling_ratio = None):
        self.dataset_dir = os.path.join(dataset_dir, "literals")
        self.normalization = normalization
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
        if self.normalization == "z-norm":
            stats = df.groupby("rel_idx")["tail"].agg(["mean", "std"])
            self.normalization_params = stats.to_dict(orient="index")
            df["tail_norm"] = df.groupby("rel_idx")["tail"].transform(
                lambda x: (x - x.mean()) / x.std()
            )

        elif self.normalization == "min-max":
            stats = df.groupby("rel_idx")["tail"].agg(["min", "max"])
            self.normalization_params = stats.to_dict(orient="index")
            df["tail_norm"] = df.groupby("rel_idx")["tail"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )

        else:
            print(" No normalization applied.")
            df["tail_norm"] = df["tail"]
            self.normalization_params = None

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

    def get_batch(
        self, entity_indices: torch.Tensor, multi_regression=False
    ):
        indices = torch.where(torch.isin(self.triples[:, 0], entity_indices))[0]
        ent_ids = self.triples[indices, 0]
        rel_ids = self.triples[indices, 1]
        labels = self.tails_norm[indices]

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
