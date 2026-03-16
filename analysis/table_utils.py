"""Utilities for building comparison dataframes used in paper analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import os
import json
import pandas as pd


DEFAULT_RELATION_MAPPINGS = {
    "rel_map_db15k": {
        "birthDate": "birthDate",
        "completionDate": "completionDate",
        "deathDate": "deathDate",
        "formationDate": "formationDate",
        "foundingDate": "foundingDate",
        "height": "height",
        "releaseDate": "releaseDate",
        "wgs84_pos#lat": "latitude",
        "wgs84_pos#long": "longitude",
    },
    "rel_map_mutag": {
        "http://dl-learner.org/mutagenesis#act": "mutagenesis\\#act",
        "http://dl-learner.org/mutagenesis#charge": "mutagenesis\\#charge",
        "http://dl-learner.org/mutagenesis#logp": "mutagenesis\\#logp",
        "http://dl-learner.org/mutagenesis#lumo": "mutagenesis\\#lumo",
    },
    "rel_map_fb15k": {
        "people.person.date_of_birth": "date\\_of\\_birth",
        "people.deceased_person.date_of_death": "date\\_of\\_death",
        "film.film.initial_release_date": "release\\_date",
        "organization.organization.date_founded": "org.date\\_founded",
        "location.dated_location.date_founded": "loc.date\\_founded",
        "location.geocode.latitude": "latitude",
        "location.geocode.longitude": "longitude",
        "location.location.area": "loc.area",
        "topic_server.population_number": "pop.\\_number",
        "people.person.height_meters": "height\\_meters",
        "people.person.weight_kg": "weight\\_kg",
    },
    "rel_map_yago15k": {
        "diedOnDate": "diedOnDate",
        "happenedOnDate": "happenedOnDate",
        "hasLatitude": "Latitude",
        "hasLongitude": "Longitude",
        "wasBornOnDate": "BornOnDate",
        "wasCreatedOnDate": "CreatedOnDate",
        "wasDestroyedOnDate": "DestroyedOnDate",
    },
    "rel_map_yago15k_short": {
        "diedOnDate": "diedOnDate",
        "happenedOnDate": "happenedOnDate",
        "hasLatitude": "Latitude",
        "hasLongitude": "Longitude",
        "wasBornOnDate": "BornOnDate",
        "wasCreatedOnDate": "CreatedOn",
        "wasDestroyedOnDate": "DestroyedOn",
    },
}


def load_relation_mappings(json_file_path=None) -> dict:
    """Load relation mappings from in-memory defaults."""
    return {k: v.copy() for k, v in DEFAULT_RELATION_MAPPINGS.items()}


def _resolve_mapping(
    relation_mappings: dict,
    dataset_name: str | None = None,
    rel_mapping_name: str | None = None,
    rel_mapping: dict | None = None,
) -> dict | None:
    if rel_mapping_name and rel_mapping_name in relation_mappings:
        return relation_mappings[rel_mapping_name]
    if rel_mapping:
        return rel_mapping
    if dataset_name:
        if dataset_name == "DB15K":
            return relation_mappings["rel_map_db15k"]
        if dataset_name == "FB15k-237":
            return relation_mappings["rel_map_fb15k"]
        if dataset_name == "YAGO15k":
            return relation_mappings["rel_map_yago15k"]
        if dataset_name == "Mutagenesis":
            return relation_mappings["rel_map_mutag"]
    return None


def _extract_metric_mean(df: pd.DataFrame, metric: str) -> pd.Series:
    """
    Extract metric mean from literal-eval dataframe.

    Supports:
    - aggregated files: {metric}_mean
    - n-run files: {metric}_run_*
    - single-run files: {metric}
    """
    mean_col = f"{metric}_mean"
    if mean_col in df.columns:
        return df[mean_col]

    run_cols = [col for col in df.columns if col.startswith(f"{metric}_run_")]
    if run_cols:
        return df[run_cols].mean(axis=1)

    if metric in df.columns:
        return df[metric]

    raise ValueError(f"Could not find columns for metric '{metric}' in literal eval results.")


def _map_relation_column(df: pd.DataFrame, mapping: dict | None) -> pd.DataFrame:
    """Apply mapping to relation column, but keep original if already mapped or missing."""
    if mapping is None or "relation" not in df.columns:
        return df
    df = df.copy()
    df["relation"] = df["relation"].map(mapping).fillna(df["relation"])
    return df


def build_comparison_dataframe(
    dataset_name: str,
    rel_mapping_name: str | None = None,
    rel_mapping: dict | None = None,
    relation_mappings: dict | None = None,
    use_scientific_notation_relations: Iterable[str] | None = None,
    include_rmse: bool = False,
    baseline_source: str = "approaches",
    include_local_global: bool = True,
) -> pd.DataFrame:
    """Build the formatted comparison dataframe for LitEm and selected baseline source.

    baseline_source options:
    - "approaches"
    - "linear_regression"
    - "both"
    - "none"
    """
    relation_mappings = relation_mappings or load_relation_mappings()
    approaches_supported_datasets = {"FB15k-237", "YAGO15k"}
    if dataset_name not in approaches_supported_datasets and baseline_source in {"approaches", "both"}:
        print(
            f"Baseline not available for dataset '{dataset_name}' "
            "(approaches supported only for FB15k-237 and YAGO15k)."
        )
        if baseline_source == "approaches":
            baseline_source = "none"
        else:
            baseline_source = "linear_regression"

    mapping = _resolve_mapping(
        relation_mappings=relation_mappings,
        dataset_name=dataset_name,
        rel_mapping_name=rel_mapping_name,
        rel_mapping=rel_mapping,
    )

    local_global_df = pd.read_csv(f"Stats/{dataset_name}_LOCAL_GLOBAL.csv", sep=",")
    local_global_df = _map_relation_column(local_global_df, mapping)
    relation_df = local_global_df[["relation"]].copy()

    if baseline_source == "approaches":
        baseline_df = pd.read_csv(f"Stats/{dataset_name}_approaches.csv", sep=",")
        baseline_df = _map_relation_column(baseline_df, mapping)
    elif baseline_source == "linear_regression":
        baseline_df = pd.read_csv(f"Stats/{dataset_name}_linear_regression.csv", sep=",")
        baseline_df = _map_relation_column(baseline_df, mapping)
    elif baseline_source == "both":
        approaches_df = pd.read_csv(f"Stats/{dataset_name}_approaches.csv", sep=",")
        linear_reg_df = pd.read_csv(f"Stats/{dataset_name}_linear_regression.csv", sep=",")
        approaches_df = _map_relation_column(approaches_df, mapping)
        linear_reg_df = _map_relation_column(linear_reg_df, mapping)
        overlapping = set(approaches_df.columns).intersection(set(linear_reg_df.columns)) - {"relation"}
        if overlapping:
            approaches_df = approaches_df.rename(columns={c: f"{c}_approaches" for c in overlapping})
            linear_reg_df = linear_reg_df.rename(columns={c: f"{c}_linear" for c in overlapping})
        baseline_df = pd.merge(approaches_df, linear_reg_df, on="relation", how="inner")
    elif baseline_source == "none":
        baseline_df = relation_df.copy()
    else:
        raise ValueError(
            "baseline_source must be one of: ['approaches', 'linear_regression', 'both', 'none']"
        )

    litem_df = pd.read_csv(
        f"Experiments/Literals/{dataset_name}/TransE_128_mlp/lit_eval_results.csv", sep=","
    )
    litem_base_df = pd.read_csv(
        f"Experiments/Literals/{dataset_name}/TransE_128_mlp_no_res/lit_eval_results.csv", sep=","
    )
    litem_df = _map_relation_column(litem_df, mapping)
    litem_base_df = _map_relation_column(litem_base_df, mapping)

    # LitEm is formatted as mean only (no std).
    litem_df["MAE_litem_mean"] = _extract_metric_mean(litem_df, "MAE")
    litem_df["RMSE_litem_mean"] = _extract_metric_mean(litem_df, "RMSE")

    litem_df["MAE_litem"] = litem_df["MAE_litem_mean"].map(lambda val: f"{val:.3f}")
    litem_df["RMSE_litem"] = litem_df["RMSE_litem_mean"].map(lambda val: f"{val:.3f}")
    litem_df = litem_df[["relation", "MAE_litem", "RMSE_litem"]]

    litem_base_df["MAE_litem_base_mean"] = _extract_metric_mean(litem_base_df, "MAE")
    litem_base_df["RMSE_litem_base_mean"] = _extract_metric_mean(litem_base_df, "RMSE")
    litem_base_df["MAE_litem_base"] = litem_base_df["MAE_litem_base_mean"].map(
        lambda val: f"{val:.3f}"
    )
    litem_base_df["RMSE_litem_base"] = litem_base_df["RMSE_litem_base_mean"].map(
        lambda val: f"{val:.3f}"
    )
    litem_base_df = litem_base_df[["relation", "MAE_litem_base", "RMSE_litem_base"]]

    merged_df = relation_df.copy()
    if include_local_global:
        merged_df = pd.merge(merged_df, local_global_df, on="relation", how="inner")
    if baseline_source != "none":
        merged_df = pd.merge(merged_df, baseline_df, on="relation", how="inner")
    merged_df = pd.merge(merged_df, litem_base_df, on="relation", how="inner")
    final_df = pd.merge(merged_df, litem_df, on="relation", how="inner")

    pd.set_option("display.float_format", "{:.5f}".format)
    metric_pattern = "relation|MAE|RMSE" if include_rmse else "relation|MAE"
    filtered_df = final_df.loc[:, final_df.columns.str.contains(metric_pattern, case=False)].copy()
    tail_cols = ["MAE_litem_base", "MAE_litem"]
    if include_rmse:
        tail_cols.extend(["RMSE_litem_base", "RMSE_litem"])
    middle_cols = [c for c in filtered_df.columns if c not in (["relation"] + tail_cols)]
    ordered_cols = ["relation"] + middle_cols + [c for c in tail_cols if c in filtered_df.columns]
    filtered_df = filtered_df[ordered_cols]

    for col in filtered_df.columns:
        if col not in [
            "relation",
            "MAE_litem",
            "RMSE_litem",
            "MAE_litem_base",
            "RMSE_litem_base",
        ]:
            filtered_df[col] = filtered_df[col].astype("object")
            filtered_df.loc[:, col] = filtered_df[col].apply(
                lambda x: f"{x:.3f}" if pd.notnull(x) else "-"
            )

    if use_scientific_notation_relations:
        target_relations = set(use_scientific_notation_relations)
        for idx, row in filtered_df.iterrows():
            if row["relation"] in target_relations:
                for col in filtered_df.columns:
                    if col not in [
                        "relation",
                        "MAE_litem",
                        "RMSE_litem",
                        "MAE_litem_base",
                        "RMSE_litem_base",
                    ] and row[col] != "-":
                        try:
                            val = float(row[col]) if isinstance(row[col], str) else float(row[col])
                            if abs(val) > 1e3:
                                filtered_df.at[idx, col] = f"{val:.2e}"
                        except Exception:
                            continue

    return filtered_df




def build_comparison_dataframe_models(
    dataset_name: str,
    model_names: list[str],
    embedding_dim: int | str,
    mlp_type: str = "mlp",
    no_res: bool = False,
    base_exp_dir: str = "Experiments/Literals",
    include_rmse: bool = False,
    rel_mapping_name: str | None = None,
    rel_mapping: dict | None = None,
    relation_mappings: dict | None = None,
) -> pd.DataFrame:
    """
    Build a comparison dataframe across models from experiment folders.

    Expected path pattern per model:
    {base_exp_dir}/{dataset_name}_combined/{model}_{embedding_dim}/lit_results.json
    """
    if not model_names:
        raise ValueError("model_names must be a non-empty list.")

    relation_mappings = relation_mappings or load_relation_mappings()
    mapping = _resolve_mapping(
        relation_mappings=relation_mappings,
        dataset_name=dataset_name,
        rel_mapping_name=rel_mapping_name,
        rel_mapping=rel_mapping,
    )

    merged_df = None
    missing_paths = []

    for model_name in model_names:
        model_embedding_dim = embedding_dim
        if model_name == "Pykeen_RotatE":
            model_name = "RotatE"
            try:
                model_embedding_dim = int(embedding_dim) // 2
            except (TypeError, ValueError):
                model_embedding_dim = embedding_dim

        run_name = f"{model_name}_{model_embedding_dim}_{mlp_type}"
        if no_res:
            run_name += "_no_res"
        result_path = Path(base_exp_dir) / dataset_name / run_name / "lit_eval_results.csv"
        if not result_path.exists():
            missing_paths.append(str(result_path))
            continue

        model_df = pd.read_csv(result_path, sep=",")
        model_df = _map_relation_column(model_df, mapping)
        if dataset_name == "FB15k-237" and "relation" in model_df.columns:
            excluded_relations = {
                "location.location.area",
                "topic_server.population_number",
                "loc.area",
                "pop.\\_number",
            }
            model_df = model_df[~model_df["relation"].isin(excluded_relations)]
        cols = ["relation"]
        display_model_name = "RotatE" if model_name == "Pykeen_RotatE" else model_name
        label = f"{display_model_name}_{model_embedding_dim}_{mlp_type}"
        if no_res:
            label += "_no_res"

        model_df[f"MAE_{label}"] = _extract_metric_mean(model_df, "MAE")
        cols.append(f"MAE_{label}")
        if include_rmse:
            model_df[f"RMSE_{label}"] = _extract_metric_mean(model_df, "RMSE")
            cols.append(f"RMSE_{label}")
        for col in cols:
            if col != "relation":
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                model_df[col] = model_df[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
        model_df = model_df[cols]

        if merged_df is None:
            merged_df = model_df
        else:
            merged_df = pd.merge(merged_df, model_df, on="relation", how="inner")

    if merged_df is None:
        raise FileNotFoundError("No lit_eval_results.csv found for the requested models.")

    if missing_paths:
        print(f"Warning: skipped models with missing results: {missing_paths}")

    metric_cols = sorted([c for c in merged_df.columns if c != "relation"])
    return merged_df[["relation"] + metric_cols]

def build_comparison_dataframe_models_combined(
    dataset_name: str,
    model_names: list[str],
    embedding_dim: int | str,
    mlp_type: str = "mlp",
    no_res: bool = False,
    base_exp_dir: str = "Experiments/KGE_Combined",
    include_rmse: bool = False,
    rel_mapping_name: str | None = None,
    rel_mapping: dict | None = None,
    relation_mappings: dict | None = None,
) -> pd.DataFrame:
    """
    Build a comparison dataframe across models from experiment folders.

    Expected path pattern per model:
    {base_exp_dir}/{dataset_name}/{model}_{embedding_dim}_{mlp_type}[ _no_res]/lit_eval_results.csv
    """
    if not model_names:
        raise ValueError("model_names must be a non-empty list.")

    relation_mappings = relation_mappings or load_relation_mappings()
    mapping = _resolve_mapping(
        relation_mappings=relation_mappings,
        dataset_name=dataset_name,
        rel_mapping_name=rel_mapping_name,
        rel_mapping=rel_mapping,
    )

    merged_df = None
    missing_paths = []

    for model_name in model_names:
        model_embedding_dim = embedding_dim
        if model_name == "Pykeen_RotatE":
            model_name = "RotatE"
            try:
                model_embedding_dim = int(embedding_dim) // 2
            except (TypeError, ValueError):
                model_embedding_dim = embedding_dim

        result_path = (
            Path(base_exp_dir)
            / f"{dataset_name}_combined"
            / f"{model_name}_{model_embedding_dim}"
            / "lit_results.json"
        )
        if not result_path.exists():
            missing_paths.append(str(result_path))
            continue

        model_df = pd.read_json(result_path)
        model_df = _map_relation_column(model_df, mapping)
        if dataset_name == "FB15k-237" and "relation" in model_df.columns:
            excluded_relations = {
                "location.location.area",
                "topic_server.population_number",
                "loc.area",
                "pop.\\_number",
            }
            model_df = model_df[~model_df["relation"].isin(excluded_relations)]
        cols = ["relation"]
        display_model_name = "RotatE" if model_name == "Pykeen_RotatE" else model_name
        label = f"{display_model_name}_{model_embedding_dim}_{mlp_type}"
        if no_res:
            label += "_no_res"

        model_df[f"MAE_{label}"] = _extract_metric_mean(model_df, "MAE")
        cols.append(f"MAE_{label}")
        if include_rmse:
            model_df[f"RMSE_{label}"] = _extract_metric_mean(model_df, "RMSE")
            cols.append(f"RMSE_{label}")
        for col in cols:
            if col != "relation":
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
                model_df[col] = model_df[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
        model_df = model_df[cols]

        if merged_df is None:
            merged_df = model_df
        else:
            merged_df = pd.merge(merged_df, model_df, on="relation", how="inner")

    if merged_df is None:
        raise FileNotFoundError("No lit_results.json found for the requested models.")

    if missing_paths:
        print(f"Warning: skipped models with missing results: {missing_paths}")

    metric_cols = sorted([c for c in merged_df.columns if c != "relation"])
    return merged_df[["relation"] + metric_cols]


def build_literal_means_with_disjoint(
    dataset_name: str,
    model_name: str = "TransE",
    embedding_dim: int | str = 128,
    mlp_type: str = "mlp",
    no_res: bool = False,
    base_exp_dir: str = "Experiments/Literals",
    rel_mapping_name: str | None = None,
    rel_mapping: dict | None = None,
    relation_mappings: dict | None = None,
    use_scientific_notation_relations: Iterable[str] | None = None,
    scientific_threshold: float = 1e3,
) -> pd.DataFrame:
    """
    Build a dataframe with relation-level MAE/RMSE means for standard and disjoint splits.

    Output columns:
    relation, MAE_mean, RMSE_mean, MAE_mean_disjoint, RMSE_mean_disjoint

    Expected path pattern:
    {base_exp_dir}/{dataset_name}/{model}_{embedding_dim}_{mlp_type}[ _no_res]/lit_eval_results.csv
    {base_exp_dir}/{dataset_name}_disjoint/{model}_{embedding_dim}_{mlp_type}[ _no_res]/lit_eval_results.csv
    """
    relation_mappings = relation_mappings or load_relation_mappings()
    mapping = _resolve_mapping(
        relation_mappings=relation_mappings,
        dataset_name=dataset_name,
        rel_mapping_name=rel_mapping_name,
        rel_mapping=rel_mapping,
    )

    run_name = f"{model_name}_{embedding_dim}_{mlp_type}"
    if no_res:
        run_name += "_no_res"

    normal_path = Path(base_exp_dir) / dataset_name / run_name / "lit_eval_results.csv"
    disjoint_path = (
        Path(base_exp_dir) / f"{dataset_name}_disjoint" / run_name / "lit_eval_results.csv"
    )

    if not normal_path.exists():
        raise FileNotFoundError(f"Missing lit_eval_results.csv: {normal_path}")
    if not disjoint_path.exists():
        raise FileNotFoundError(f"Missing lit_eval_results.csv: {disjoint_path}")

    normal_df = pd.read_csv(normal_path, sep=",")
    disjoint_df = pd.read_csv(disjoint_path, sep=",")

    normal_df = _map_relation_column(normal_df, mapping)
    disjoint_df = _map_relation_column(disjoint_df, mapping)

    normal_df["MAE_mean"] = _extract_metric_mean(normal_df, "MAE")
    normal_df["RMSE_mean"] = _extract_metric_mean(normal_df, "RMSE")
    normal_df = normal_df[["relation", "MAE_mean", "RMSE_mean"]]

    disjoint_df["MAE_mean_disjoint"] = _extract_metric_mean(disjoint_df, "MAE")
    disjoint_df["RMSE_mean_disjoint"] = _extract_metric_mean(disjoint_df, "RMSE")
    disjoint_df = disjoint_df[["relation", "MAE_mean_disjoint", "RMSE_mean_disjoint"]]

    merged_df = pd.merge(normal_df, disjoint_df, on="relation", how="inner")
    ordered_cols = [
        "relation",
        "MAE_mean",
        "RMSE_mean",
        "MAE_mean_disjoint",
        "RMSE_mean_disjoint",
    ]
    merged_df = merged_df[ordered_cols]
    for col in ordered_cols:
        if col != "relation":
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")
            merged_df[col] = merged_df[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")

    if use_scientific_notation_relations:
        target_relations = set(use_scientific_notation_relations)
        for idx, row in merged_df.iterrows():
            if row["relation"] in target_relations:
                for col in ordered_cols:
                    if col != "relation" and row[col] != "-":
                        try:
                            val = float(row[col])
                            if abs(val) > scientific_threshold:
                                merged_df.at[idx, col] = f"{val:.2e}"
                        except Exception:
                            continue
    return merged_df


def build_ablation_table(
    dataset_name: str,
    splits: Iterable[int] = (20, 40, 60, 80),
    model_name: str = "TransE",
    base_exp_dir: str = "Experiments/Ablations",
    rel_mapping_name: str | None = None,
    rel_mapping: dict | None = None,
    relation_mappings: dict | None = None,
    sort_by: str | None = None,
    descending: bool = True,
) -> pd.DataFrame:
    """
    Build a relation-level ablation table across splits.

    Output columns:
    relation, MAE_mean_{split}, RMSE_mean_{split} for each split.

    Expected path pattern:
    {base_exp_dir}/{dataset_name}_{split}/{model_name}/lit_eval_results.csv
    """
    relation_mappings = relation_mappings or load_relation_mappings()
    mapping = _resolve_mapping(
        relation_mappings=relation_mappings,
        dataset_name=dataset_name,
        rel_mapping_name=rel_mapping_name,
        rel_mapping=rel_mapping,
    )

    merged_df = None
    missing_paths = []
    splits = list(splits)

    for split in splits:
        result_path = (
            Path(base_exp_dir)
            / f"{dataset_name}_{split}"
            / model_name
            / "lit_eval_results.csv"
        )
        if not result_path.exists():
            missing_paths.append(str(result_path))
            continue

        split_df = pd.read_csv(result_path, sep=",")
        split_df = _map_relation_column(split_df, mapping)

        split_df[f"MAE_mean_{split}"] = _extract_metric_mean(split_df, "MAE")
        split_df[f"RMSE_mean_{split}"] = _extract_metric_mean(split_df, "RMSE")
        split_df = split_df[["relation", f"MAE_mean_{split}", f"RMSE_mean_{split}"]]

        if merged_df is None:
            merged_df = split_df
        else:
            merged_df = pd.merge(merged_df, split_df, on="relation", how="inner")

    if merged_df is None:
        raise FileNotFoundError("No lit_eval_results.csv found for the requested ablation splits.")

    if missing_paths:
        print(f"Warning: skipped splits with missing results: {missing_paths}")

    ordered_cols = ["relation"]
    for split in splits:
        mae_col = f"MAE_mean_{split}"
        rmse_col = f"RMSE_mean_{split}"
        if mae_col in merged_df.columns:
            ordered_cols.append(mae_col)
        if rmse_col in merged_df.columns:
            ordered_cols.append(rmse_col)

    merged_df = merged_df[ordered_cols]

    for col in ordered_cols:
        if col != "relation":
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    if sort_by is None:
        for split in splits:
            candidate = f"MAE_mean_{split}"
            if candidate in merged_df.columns:
                sort_by = candidate
                break

    if sort_by and sort_by in merged_df.columns:
        merged_df = merged_df.sort_values(by=sort_by, ascending=not descending)

    for col in ordered_cols:
        if col != "relation":
            merged_df[col] = merged_df[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")

    return merged_df


def build_ablation_delta_table(
    dataset_name: str,
    splits: Iterable[int] = (80, 60, 40, 20),
    model_name: str = "TransE",
    base_ablation_dir: str = "Experiments/Ablations",
    base_literal_dir: str = "Experiments/Literals",
    baseline_model_name: str = "TransE",
    baseline_embedding_dim: int | str = 128,
    baseline_mlp_type: str = "mlp",
    baseline_no_res: bool = False,
    rel_mapping_name: str | None = None,
    rel_mapping: dict | None = None,
    relation_mappings: dict | None = None,
    decimals: int = 2,
    descending: bool = True,
    sort_by: str | None = None,
) -> pd.DataFrame:
    """
    Build a relation-level ablation table with % deltas vs a baseline literal run.

    Output columns:
    relation, MAE_{split}, RMSE_{split} (percent change vs baseline)

    Ablation path pattern:
    {base_ablation_dir}/{dataset_name}_{split}/{model_name}/lit_eval_results.csv

    Baseline path pattern:
    {base_literal_dir}/{dataset_name}/{baseline_model_name}_{baseline_embedding_dim}_{baseline_mlp_type}[ _no_res]/lit_eval_results.csv
    """
    relation_mappings = relation_mappings or load_relation_mappings()
    mapping = _resolve_mapping(
        relation_mappings=relation_mappings,
        dataset_name=dataset_name,
        rel_mapping_name=rel_mapping_name,
        rel_mapping=rel_mapping,
    )

    baseline_run = f"{baseline_model_name}_{baseline_embedding_dim}_{baseline_mlp_type}"
    if baseline_no_res:
        baseline_run += "_no_res"
    baseline_path = Path(base_literal_dir) / dataset_name / baseline_run / "lit_eval_results.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline lit_eval_results.csv: {baseline_path}")

    baseline_df = pd.read_csv(baseline_path, sep=",")
    baseline_df = _map_relation_column(baseline_df, mapping)
    baseline_df["MAE_base"] = _extract_metric_mean(baseline_df, "MAE")
    baseline_df["RMSE_base"] = _extract_metric_mean(baseline_df, "RMSE")
    baseline_df = baseline_df[["relation", "MAE_base", "RMSE_base"]]

    merged_df = baseline_df
    missing_paths = []
    splits = list(splits)

    for split in splits:
        result_path = (
            Path(base_ablation_dir)
            / f"{dataset_name}_{split}"
            / model_name
            / "lit_eval_results.csv"
        )
        if not result_path.exists():
            missing_paths.append(str(result_path))
            continue

        split_df = pd.read_csv(result_path, sep=",")
        split_df = _map_relation_column(split_df, mapping)
        split_df[f"MAE_{split}"] = _extract_metric_mean(split_df, "MAE")
        split_df[f"RMSE_{split}"] = _extract_metric_mean(split_df, "RMSE")
        split_df = split_df[["relation", f"MAE_{split}", f"RMSE_{split}"]]

        merged_df = pd.merge(merged_df, split_df, on="relation", how="inner")

    if missing_paths:
        print(f"Warning: skipped splits with missing results: {missing_paths}")

    for split in splits:
        mae_col = f"MAE_{split}"
        rmse_col = f"RMSE_{split}"
        if mae_col in merged_df.columns:
            merged_df[mae_col] = (
                (merged_df[mae_col] - merged_df["MAE_base"]) / merged_df["MAE_base"]
            ) * 100
        if rmse_col in merged_df.columns:
            merged_df[rmse_col] = (
                (merged_df[rmse_col] - merged_df["RMSE_base"]) / merged_df["RMSE_base"]
            ) * 100

    merged_df = merged_df.drop(columns=["MAE_base", "RMSE_base"])

    ordered_cols = ["relation"]
    for split in splits:
        mae_col = f"MAE_{split}"
        rmse_col = f"RMSE_{split}"
        if mae_col in merged_df.columns:
            ordered_cols.append(mae_col)
        if rmse_col in merged_df.columns:
            ordered_cols.append(rmse_col)
    merged_df = merged_df[ordered_cols]

    for col in ordered_cols:
        if col != "relation":
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    if sort_by is None:
        for split in splits:
            candidate = f"MAE_{split}"
            if candidate in merged_df.columns:
                sort_by = candidate
                break
    if sort_by and sort_by in merged_df.columns:
        merged_df = merged_df.sort_values(by=sort_by, ascending=not descending)

    fmt = f"{{:.{decimals}f}}"
    for col in ordered_cols:
        if col != "relation":
            merged_df[col] = merged_df[col].map(lambda x: fmt.format(x) if pd.notnull(x) else "-")

    return merged_df


def _format_metric(value):
    if value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return value


def _insert_dataset_spacer_columns(df_in, datasets, metrics):
    if len(datasets) <= 1:
        return df_in

    reordered_columns = ["Model"]
    for index, dataset in enumerate(datasets):
        reordered_columns.extend([f"{metric}_{dataset}" for metric in metrics])
        if index != len(datasets) - 1:
            spacer_name = f"Spacer_{index + 1}"
            df_in[spacer_name] = ""
            reordered_columns.append(spacer_name)
    return df_in[reordered_columns]


def _bold_blockwise_maxima(df_in, block_size, metric_columns):
    df_out = df_in.copy()
    if block_size <= 0:
        return df_out
    for column in metric_columns:
        df_out[column] = df_out[column].astype(object)

    for start in range(0, len(df_out), block_size):
        block = df_out.iloc[start:start + block_size]
        if block.empty:
            continue
        for column in metric_columns:
            numeric_values = pd.to_numeric(block[column], errors="coerce")
            if numeric_values.isna().all():
                continue
            best_value = numeric_values.max()
            best_rows = numeric_values[numeric_values == best_value].index
            for row_index in best_rows:
                df_out.at[row_index, column] = f"\\textbf{{{best_value:.3f}}}"
    return df_out

import re
def dataframe_to_latex(df, **kwargs):
    latex = df.to_latex(escape=False, **kwargs)
    return re.sub(r"^\\midrule(?:\s*&\s*)*\\\\$", r"\\midrule", latex, flags=re.MULTILINE)


def _resolve_combined_eval_path(
    dataset: str,
    model: str,
    embedding_dim: int,
    strategy: str = "auto",
):
    strategy_key = str(strategy).strip().lower()
    if strategy_key == "auto":
        strategy_order = ("grad", "harmonic", "uncertainty", "legacy")
    else:
        strategy_order = (strategy_key,)

    def _candidate_path(curr_strategy: str):
        if curr_strategy in {"legacy", "default", "base"}:
            folder = f"{dataset}_combined"
        else:
            folder = f"{dataset}_combined_{curr_strategy}"
        return f"Experiments/KGE_Combined/{folder}/{model}_{embedding_dim}/eval_report.json"

    candidates = [_candidate_path(curr_strategy) for curr_strategy in strategy_order]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def get_kge_combined_table(
    dataset1: str,
    dataset2: str = None,
    dataset3: str = None,
    embedding_dim: int = 64,
    approaches=None,
):
    """
    Generate a test-only comparison table for up to three datasets.

    Rows are grouped by base model and expanded into four variants:
    - base KGE
    - combined training (+LitEM)
    - LiteralE
    - KBLN

    A single shared embedding dimension is enforced for every dataset
    and every approach in the table.

    The `approaches` argument selects which rows to include. Valid values:
    - "base"
    - "kbln"
    - "literale"
    - "litem"

    Columns are:
    - Model
    - for each dataset: MRR, H@1, H@3, H@10
    """

    datasets = [dataset for dataset in (dataset1, dataset2, dataset3) if dataset]
    if not datasets:
        raise ValueError("At least one dataset must be provided.")
    if len(datasets) > 3:
        raise ValueError("A maximum of three datasets is supported.")
    if embedding_dim is None:
        raise ValueError("`embedding_dim` must be set to one shared value for the whole table.")

    embedding_dim = int(embedding_dim)

    base_models = ["ComplEx", "DeCaL", "DistMult", "DualE", "Keci", "OMult", "QMult" , "TransE"]
    metrics = ("MRR", "H@1", "H@3", "H@10")
    variant_map = {
        "base": ("", "Experiments/KGE/{dataset}/{model}_{embedding_dim}/eval_report.json"),
        "kbln": ("+KBLN", "Experiments/KGE_KBLN/{dataset}/{model}_{embedding_dim}/eval_report.json"),
        "literale": ("+LiteralE", "Experiments/KGE_LiteralE/{dataset}/{model}_{embedding_dim}/eval_report.json"),
        "litem": ("+LitEM", None),
    }
    if approaches is None:
        approaches = ["base", "kbln", "literale", "litem"]

    normalized_approaches = []
    for approach in approaches:
        key = str(approach).strip().lower()
        if key not in variant_map:
            raise ValueError(
                f"Unknown approach `{approach}`. Valid options are: {', '.join(variant_map)}"
            )
        normalized_approaches.append(key)

    if not normalized_approaches:
        raise ValueError("At least one approach must be selected.")

    variants = tuple((key, variant_map[key][0], variant_map[key][1]) for key in normalized_approaches)

    columns = ["Model"]
    for dataset in datasets:
        columns.extend([f"{metric}_{dataset}" for metric in metrics])

    def get_eval_path(dataset, model, approach_key, path_template):
        if approach_key == "litem":
            return _resolve_combined_eval_path(
                dataset=dataset,
                model=model,
                embedding_dim=embedding_dim,
                strategy="auto",
            )
        return path_template.format(
            dataset=dataset,
            model=model,
            embedding_dim=embedding_dim,
        )

    def validate_shared_embedding_dim():
        mismatches = []
        for dataset in datasets:
            for model in base_models:
                for approach_key, suffix, path_template in variants:
                    eval_path = get_eval_path(dataset, model, approach_key, path_template)
                    if os.path.exists(eval_path):
                        continue

                    variant_dir = os.path.dirname(os.path.dirname(eval_path))
                    if not os.path.isdir(variant_dir):
                        continue

                    prefix = f"{model}_"
                    available_dims = sorted(
                        entry[len(prefix):]
                        for entry in os.listdir(variant_dir)
                        if entry.startswith(prefix)
                        and os.path.isdir(os.path.join(variant_dir, entry))
                        and entry[len(prefix):].isdigit()
                    )
                    if available_dims and str(embedding_dim) not in available_dims:
                        mismatches.append(
                            f"{dataset} / {model}{suffix}: requested {embedding_dim}, available {', '.join(available_dims)}"
                        )

        if mismatches:
            raise ValueError(
                "Embedding dimension must be identical across all datasets and approaches. "
                + "Mismatches found: "
                + "; ".join(mismatches)
            )

    validate_shared_embedding_dim()

    def extract_test_metrics(eval_path):
        if not os.path.exists(eval_path):
            return [""] * len(metrics)

        with open(eval_path, "r") as handle:
            eval_data = json.load(handle)

        test_metrics = eval_data.get("Test", {})
        return [test_metrics.get(metric, "") for metric in metrics]

    data_rows = []
    for base_model in base_models:
        for variant_index, (approach_key, suffix, path_template) in enumerate(variants):
            row_label = base_model if variant_index == 0 else suffix
            row = [row_label]

            for dataset in datasets:
                eval_path = get_eval_path(dataset, base_model, approach_key, path_template)
                row.extend(extract_test_metrics(eval_path))
            data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=columns)
    metric_columns = [column for column in df.columns if column != "Model"]
    df = _bold_blockwise_maxima(df, len(variants), metric_columns)
    df = _insert_dataset_spacer_columns(df, datasets, metrics)

    for column in df.columns[1:]:
        df[column] = df[column].map(_format_metric)

    def insert_midrules(df_in):
        new_rows = []

        for index, base_model in enumerate(base_models):
            start_row = index * len(variants)
            block = df_in.iloc[start_row:start_row + len(variants)].copy()
            new_rows.append(block)
            if index != len(base_models) - 1:
                new_rows.append(
                    pd.DataFrame(
                        [["\\midrule"] + [""] * (len(df_in.columns) - 1)],
                        columns=df_in.columns,
                    )
                )

        if not new_rows:
            return df_in
        return pd.concat(new_rows, ignore_index=True)

    return insert_midrules(df)