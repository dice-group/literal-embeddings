import os

import pandas as pd
import torch


class LiteralData:
    def __init__(self, dataset_dir: str, ent_idx, filter_df=False, top_k = 10):

        self.train_file_path = os.path.join(dataset_dir, "train.txt")
        self.test_file_path = os.path.join(dataset_dir, "test.txt")
        self.val_file_path = os.path.join(dataset_dir, "valid.txt")
        self.ent_idx = {value: idx for idx, value in ent_idx["entity"].items()}

        df = pd.read_csv(
            self.train_file_path,
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )

        if filter_df:
            top_k_rels = df["relation"].value_counts().nlargest(top_k).index

            # Step 2: Filter the DataFrame to include only rows with these top 10 items
            df = df[df["relation"].isin(top_k_rels)]
            self.train_rels = top_k_rels
        else:
            self.train_rels = df["relation"].unique().tolist()

        df = df[df["head"].isin(self.ent_idx)]
        df["tail"] = df["tail"].astype(float)

        self.unique_relations = df["relation"].unique()
        self.num_data_properties = len(self.unique_relations)
        
        self.data_property_to_idx = {
            relation: idx for idx, relation in enumerate(self.unique_relations)
        }

        df["head_idx"] = df["head"].map(self.ent_idx)
        df["rel_idx"] = df["relation"].map(self.data_property_to_idx)

        # Calculate normalization parameters for each relation group
        self.normalization_params = {}
        for relation in self.unique_relations:
            group_data = df.loc[df["relation"] == relation, "tail"]
            mean = group_data.mean()
            std = group_data.std()
            self.normalization_params[relation] = {"mean": mean, "std": std}

        # Normalize the tail values using the stored parameters
        df["normalized_tail"] = (
            df["tail"] - df.groupby("relation")["tail"].transform("mean")
        ) / df.groupby("relation")["tail"].transform("std")
        self.train_df = df
        # Pivot to create a new DataFrame where rows are indexed by 'head' and columns are literals
        normalized_scores = df.pivot_table(
            index="head", columns="relation", values="normalized_tail", aggfunc="first"
        ).reset_index()

        # Handle missing heads efficiently
        missing_heads = set(self.ent_idx.keys()) - set(df["head"])
        if missing_heads:
            missing_rows = pd.DataFrame({"head": list(missing_heads)})

            # Concatenate missing rows with normalized_scores
            normalized_scores = pd.concat(
                [normalized_scores, missing_rows], ignore_index=True
            )

        # Fill missing values with 0
        normalized_scores.fillna(0, inplace=True)

        # Add 'head_index' column
        normalized_scores["head_index"] = normalized_scores["head"].map(self.ent_idx)

        # Get column names dynamically from self.data_property_to_idx
        columns = ["head_index"] + list(self.data_property_to_idx.keys())

        # Reorder and sort by 'head_index'
        self.normalized_df = (
            normalized_scores[columns]
            .sort_values(by="head_index")
            .reset_index(drop=True)
        )

        # Convert selected columns to PyTorch tensor
        self.lit_value_tensor = torch.tensor(
            self.normalized_df[self.data_property_to_idx.keys()].values,
            dtype=torch.float32,
        )

        self.lit_bool_tensor = torch.tensor(
            (self.normalized_df[self.data_property_to_idx.keys()].values != 0).astype(
                int
            ),
            dtype=torch.int32,
        )
