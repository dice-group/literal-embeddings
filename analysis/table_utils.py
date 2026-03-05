"""Utilities for building comparison dataframes used in paper analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

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
        if dataset_name == "mutagenesis":
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
        f"Experiments/Literals/{dataset_name}/TransE_100_mlp/lit_eval_results.csv", sep=","
    )
    litem_base_df = pd.read_csv(
        f"Experiments/Literals/{dataset_name}/TransE_100_mlp_no_res/lit_eval_results.csv", sep=","
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
        run_name = f"{model_name}_{embedding_dim}_{mlp_type}"
        if no_res:
            run_name += "_no_res"
        result_path = Path(base_exp_dir) / dataset_name / run_name / "lit_eval_results.csv"
        if not result_path.exists():
            missing_paths.append(str(result_path))
            continue

        model_df = pd.read_csv(result_path, sep=",")
        model_df = _map_relation_column(model_df, mapping)
        cols = ["relation"]
        label = run_name

        model_df[f"MAE_{label}"] = _extract_metric_mean(model_df, "MAE")
        cols.append(f"MAE_{label}")
        if include_rmse:
            model_df[f"RMSE_{label}"] = _extract_metric_mean(model_df, "RMSE")
            cols.append(f"RMSE_{label}")
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
