import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_LOCAL_GLOBAL(df_entity_triples, df_train, df_test):
    """
    Evaluates the performance of local and global average prediction
    on an incomplete knowledge graph.

    Args:
        df_entity_triples: DataFrame containing all triples (head, relation, tail).
        df_train: DataFrame containing training data.
        df_test: DataFrame containing incomplete triples
                      to Evaluate LOCAL and GLOBAL values.

    Returns:
        A DataFrame containing Mean Absolute Error (MAE) and Root Mean Sqaured Error (RMSE)
        for 'LOCAL' and 'GLOBAL' predictions per relation.
    """
    # Precompute property lookup and global averages
    property_dict = df_train.pivot(index="head", columns="relation", values="tail")
    global_averages = df_train.groupby("relation")["tail"].mean()

    # Precompute neighbor dictionary
    neighbor_dict = df_entity_triples.groupby("head")["tail"].apply(list).to_dict()

    def calculate_local_value(row):
        node = row["head"]
        relation = row["relation"]

        # If node is not in neighbor_dict, return the global average for the relation
        if node not in neighbor_dict:
            return global_averages.get(relation, float("nan"))

        # Retrieve neighbor nodes
        neighbor_nodes = neighbor_dict[node]

        # Retrieve neighbor values for the relation
        if relation in property_dict.columns:
            neighbor_values = property_dict.loc[
                property_dict.index.intersection(neighbor_nodes), relation
            ].dropna()

            if not neighbor_values.empty:
                return neighbor_values.mean()

        # Fallback to global average
        return global_averages.get(relation, float("nan"))

    # Calculate LOCAL predictions using vectorized operations
    df_test["LOCAL"] = df_test.apply(calculate_local_value, axis=1)

    # Calculate GLOBAL predictions directly from precomputed averages
    df_test["GLOBAL"] = df_test["relation"].map(global_averages)

    # Calculate MAE and RMSE per relation
    mae_per_relation = df_test.groupby("relation").apply(
        lambda group: pd.Series(
            {
                "MAE_GLOBAL": mean_absolute_error(group["tail"], group["GLOBAL"]),
                "RMSE_GLOBAL": root_mean_squared_error(group["tail"], group["GLOBAL"]),
                "MAE_LOCAL": mean_absolute_error(group["tail"], group["LOCAL"]),
                "RMSE_LOCAL": root_mean_squared_error(group["tail"], group["LOCAL"]),
            },
        ),
        include_groups=False,
    )

    return mae_per_relation


import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Check if the provided name is  for a valid KG."
    )
    parser.add_argument(
        "--kg",
        type=str,
        help="Name of the dataset with literal files.",
        default="FB15k-237",
    )

    # Parse the arguments
    args = parser.parse_args()

    if args.kg == "FB15k-237":
        df_triples = pd.read_csv(
            "KGs/FB15k-237-lit/FB15K-237_EntityTriples.txt",
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )
        df_lits_train = pd.read_csv(
            "KGs/FB15k-237-lit/train.txt",
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )
        df_lits_test = pd.read_csv(
            "KGs/FB15k-237-lit/test.txt",
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )
        global_local_df = evaluate_LOCAL_GLOBAL(
            df_entity_triples=df_triples,
            df_train=df_lits_train,
            df_test=df_lits_test,
        )
        global_local_df.to_csv("Stats/FB15k-237-lit_LOCAL_GLOBAL.csv")
    elif args.kg == "YAGO10-plus":
        df_triples = pd.read_csv(
            "KGs/YAGO10-plus/entity_triples.txt",
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )
        df_lits_train = pd.read_csv(
            "KGs/YAGO10-plus/train.txt",
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )
        df_lits_test = pd.read_csv(
            "KGs/YAGO10-plus/test.txt",
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )
        global_local_df = evaluate_LOCAL_GLOBAL(
            df_entity_triples=df_triples,
            df_train=df_lits_train,
            df_test=df_lits_test,
        )
        global_local_df.to_csv("Stats/YAGO10-plus_LOCAL_GLOBAL.csv")
    else:
        print(
            f"The name '{args.kg}' is not valid. Please use 'FB15k-237' or 'YAGO10-plus'."
        )
        exit()


if __name__ == "__main__":
    main()
