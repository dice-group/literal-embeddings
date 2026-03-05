"""LaTeX rendering helpers for comparison tables."""

from __future__ import annotations

import pandas as pd
import re


def _to_float(value: object) -> float:
    if value == "-":
        return float("inf")
    if isinstance(value, str):
        return float(value.replace("\\textbf{", "").replace("}", ""))
    return float(value)


def _metric_tag(column_name: str) -> str | None:
    """Return metric tag ('MAE' or 'RMSE') for a column name."""
    tokens = [tok for tok in re.split(r"[^A-Za-z0-9]+", column_name.upper()) if tok]
    if "RMSE" in tokens:
        return "RMSE"
    if "MAE" in tokens:
        return "MAE"
    # Fallback for names like LR_MAE, MAE_litem, RMSE_litem_base
    upper = column_name.upper()
    if "RMSE" in upper:
        return "RMSE"
    if "MAE" in upper:
        return "MAE"
    return None


def _bold_row_minima(df: pd.DataFrame) -> pd.DataFrame:
    excluded_cols = {"relation"}
    styled_df = df.copy()

    for idx, row in styled_df.iterrows():
        # Bold minima separately for MAE and RMSE columns.
        for metric in ("MAE", "RMSE"):
            metric_cols = [
                col
                for col in styled_df.columns
                if col not in excluded_cols and _metric_tag(col) == metric
            ]
            if not metric_cols:
                continue

            metric_vals = []
            for col in metric_cols:
                try:
                    metric_vals.append(_to_float(row[col]))
                except Exception:
                    metric_vals.append(float("inf"))

            min_val = min(metric_vals) if metric_vals else float("inf")
            if min_val == float("inf"):
                continue

            for col in metric_cols:
                try:
                    current_val = _to_float(row[col])
                    if current_val == min_val:
                        if isinstance(row[col], str) and row[col].startswith("\\textbf{"):
                            continue
                        styled_df.at[idx, col] = f"\\textbf{{{row[col]}}}"
                except Exception:
                    continue

    return styled_df


def dataframe_to_latex(df: pd.DataFrame) -> str:
    """Render a comparison dataframe to LaTeX."""
    return _bold_row_minima(df).to_latex(index=False, escape=False)
