import os
import sys
# sys.path.append('..')

import pandas as pd
import torch

from src.static_funcs import load_model_components


def calculate_abalation_scores(exp_path: str):
    model_components = load_model_components(kge_path=exp_path)
    if model_components is None:
        print("Failed to load model components.")
        return
    
    kge_model = model_components.model
    configs = model_components.config
    entity_to_idx = model_components.entity_to_idx
    relation_to_idx = model_components.relation_to_idx
    
    test_kg_dir = configs["dataset_dir"] + "/test.txt"

    # process test set and map to idx
    test_df = pd.read_csv(
        test_kg_dir,
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
    )
    test_df_isa = test_df[test_df["relation"] == "/m/is_a"].reset_index(drop=True)
    test_df_isa["head_idx"] = test_df_isa["head"].map(entity_to_idx)
    test_df_isa["rel_idx"] = test_df_isa["relation"].map(relation_to_idx)
    test_df_isa["tail_idx"] = test_df_isa["tail"].map(entity_to_idx)

    # Create tensor from DataFrame
    triples = torch.tensor(
        test_df_isa[["head_idx", "rel_idx", "tail_idx"]].values, dtype=torch.long
    )

    # Forward pass with no gradient computation
    with torch.no_grad():
        test_df_isa["ranks"] = torch.sigmoid(kge_model(triples)).tolist()

    test_df_literals = pd.read_csv(
        "KGs/Synthetic/numerical_literals.txt",
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
    )
    # Filter and merge data
    high_df = test_df_isa[test_df_isa["tail"] == "/m/high"][["head", "ranks"]]
    low_df = test_df_isa[test_df_isa["tail"] == "/m/low"][["head", "ranks"]]
    merged_df = pd.merge(high_df, low_df, on="head", suffixes=("_high", "_low"))

    # Create predicted class column
    merged_df["predicted"] = (merged_df["ranks_high"] > merged_df["ranks_low"]).map(
        {True: "high", False: "low"}
    )

    # Prepare test literals data
    test_df_literals["class"] = (test_df_literals["tail"].astype(float) > 0.5).map(
        {True: "high", False: "low"}
    )

    # Merge with test literals data
    merged = merged_df.merge(test_df_literals[["head", "class"]], on="head", how="left")

    # Calculate accuracy
    true_high = (merged["predicted"] == "high") & (merged["class"] == "high")
    true_low = (merged["predicted"] == "low") & (merged["class"] == "low")
    accuracy = (true_high.sum() + true_low.sum()) / len(merged)

    return configs["model"], accuracy


def evaluate_ablations(return_df=False):
    
    original_path = "Experiments/Ablations/Synthetic"
    random_path = "Experiments/Ablations/Synthetic_random"
    # Step 1: Loop through all folders under actual and random paths
    original_scores = {}
    for folder in os.listdir(original_path):
        model_dir = os.path.join(original_path, folder)
        if os.path.isdir(model_dir):
            model_name, score = calculate_abalation_scores(model_dir)
            original_scores[folder] = score

    random_scores = {}
    for folder in os.listdir(random_path):
        model_dir = os.path.join(random_path, folder)
        if os.path.isdir(model_dir):
            model_name, score = calculate_abalation_scores(model_dir)
            random_scores[folder] = score

    # Step 2: Create DataFrame
    # Merge both sets using model name as the key
    all_models = set(original_scores.keys()).union(random_scores.keys())

    rows = []
    for model in all_models:
        rows.append(
            {
                "model": model,
                "acc_org": original_scores.get(model),
                "acc_rand": random_scores.get(model),
            }
        )

    df = pd.DataFrame(rows)

    # Optional: sort by accuracy or model name
    df = df.sort_values(by="model").reset_index(drop=True)
    storage_path = "Stats"
    os.makedirs(storage_path, exist_ok=True)

    df.to_csv(f'{storage_path}/ablation_scores.csv', sep="\t", index=False)
    print(df)
    return df if return_df else None


if __name__ == "__main__":
    evaluate_ablations()
