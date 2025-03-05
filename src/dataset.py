import os

import pandas as pd
import torch
from dicee.knowledge_graph_embeddings import KGE


class LiteralData:
    def __init__(
        self, dataset_dir: str, ent_idx, filter_df=False, normalization="z-norm"
    ):
        self.dataset_dir = dataset_dir
        self.file_paths = {
            "train": os.path.join(self.dataset_dir, "train.txt"),
            "test": os.path.join(self.dataset_dir, "test.txt"),
            "val": os.path.join(self.dataset_dir, "val.txt"),
        }

        self.normalization_params = {}
        self.normalization = normalization
        if isinstance(ent_idx, dict):
            self.entity_to_idx = ent_idx
        else:
            self.entity_to_idx = {
                value: idx for idx, value in ent_idx["entity"].items()
            }
        self.num_entities = len(self.entity_to_idx)
        self.preprocess_train()

    def preprocess_train(self):
        train_file_path = self.file_paths.get("train", None)
        assert os.path.exists(
            train_file_path
        ), f"The path {train_file_path} does not exist."
        df = pd.read_csv(
            train_file_path,
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )
        self.data_property_to_idx = {
            relation: idx for idx, relation in enumerate(df["relation"].unique())
        }
        self.num_data_properties = len(self.data_property_to_idx)
        df = df[df["head"].isin(self.entity_to_idx)]
        df["head_idx"] = df["head"].map(self.entity_to_idx)
        df["rel_idx"] = df["relation"].map(self.data_property_to_idx)
        self.normalize(df)

        # Compute mean and std for each rel_idx in a vectorized manner

        self.train_data = {
            "triples": torch.tensor(
                df[["head_idx", "rel_idx"]].values, dtype=torch.long
            ),
            "tails": torch.tensor(df[["tail"]].values, dtype=torch.float),
            "tails_norm": torch.tensor(df["tail_norm"].values, dtype=torch.float32),
        }
        print("Training data Created")

    def get_df(self, split="test", norm=False):
        file_path = self.file_paths.get(split, None)

        if file_path is None or not os.path.isfile(file_path):
            raise ValueError(
                f"Invalid data split or path does not lead to a file: {file_path}"
            )

        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )
        df["head_idx"] = df["head"].map(self.entity_to_idx)
        df["rel_idx"] = df["relation"].map(self.data_property_to_idx)
        if norm:
            df = self.normalize(df)
        return df

    def normalize(self, df):
        if self.normalization == "z-norm":
            normalization_params_df = df.groupby("rel_idx")["tail"].agg(["mean", "std"])

            # Convert to dictionary if needed (optional)
            self.normalization_params = normalization_params_df.to_dict(orient="index")
            df["tail_norm"] = df.groupby("rel_idx")["tail"].transform(
                lambda x: (x - x.mean()) / x.std()
            )
        elif self.normalization == "min-max":
            normalization_params_df = df.groupby("rel_idx")["tail"].agg(["min", "max"])

            # Convert to dictionary if needed (optional)
            self.normalization_params = normalization_params_df.to_dict(orient="index")
            df["tail_norm"] = df.groupby("rel_idx")["tail"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
        return df

    def get_batch(self, entity_idx: torch.tensor, multi_regression=True):
        # Extract head and relation IDs
        # Get row indices where the i-th column contains any of the target values
        indices = torch.where(torch.isin(self.train_data["triples"][:, 0], entity_idx))[
            0
        ]
        entids = self.train_data["triples"][indices, 0]
        rels = self.train_data["triples"][indices, 1]
        labels = self.train_data["tails_norm"][indices]

        if multi_regression:
            # Create a zero tensor of shape [num_samples, num_relations]
            num_samples = entids.shape[0]  # Number of rows in dataset
            y_true = torch.full(
                (num_samples, self.num_data_properties), -9, dtype=torch.float32
            )  # Match dtype with `v`

            # Assign tail values at the correct relation indices per row
            y_true[torch.arange(num_samples), rels] = labels
        else:
            y_true = labels
        return entids, rels, y_true

    def get_ea_encoding(self):
        # Encoding tensor
        ea_pair = torch.full(
            (self.num_entities, self.num_data_properties), 0, dtype=torch.float32
        )
        e = self.train_data["triples"][:, 0].long()
        a = self.train_data["triples"][:, 1].long()
        ea_pair[e, a] = 1
        return ea_pair
