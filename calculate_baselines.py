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
    #df_train = df_train.drop_duplicates()

    # Precompute property lookup and global averages
    property_dict = df_train.pivot_table(index="head", columns="relation", values="tail", aggfunc='first')
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


def calculate_baselines(dataset_name =None):
    dataset_path = f'KGs/{dataset_name}'
    df_triples = pd.read_csv(
            f"{dataset_path}/{dataset_name}_EntityTriples.txt",
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
        )
    df_lits_train = pd.read_csv(
        f"{dataset_path}/literals/train.txt",
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
    )
    df_lits_test = pd.read_csv(
        f"{dataset_path}/literals/test.txt",
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
    )
    global_local_df = evaluate_LOCAL_GLOBAL(
            df_entity_triples=df_triples,
            df_train=df_lits_train,
            df_test=df_lits_test,
        )
    global_local_df.to_csv(f"Stats/{dataset_name}_LOCAL_GLOBAL.csv")




def main():
    dataset_names = ["FB15k-237", "DB15K", "YAGO15k", "mutagenesis"]
    for dataset_name in dataset_names:
        calculate_baselines(dataset_name=dataset_name)
        print(f"Baseline calculation completed for{dataset_name}")

if __name__ == "__main__":
    main()
