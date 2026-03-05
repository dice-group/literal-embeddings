"""LaTeX rendering helpers for comparison tables."""

from __future__ import annotations

import pandas as pd


def _to_float(value: object) -> float:
    if value == "-":
        return float("inf")
    if isinstance(value, str):
        return float(value.replace("\\textbf{", "").replace("}", ""))
    return float(value)


def _bold_row_minima(df: pd.DataFrame) -> pd.DataFrame:
    excluded_cols = {
        "relation",
        "MAE_litem",
        "RMSE_litem",
        "MAE_litem_base",
        "RMSE_litem_base",
    }
    styled_df = df.copy()

    for idx, row in styled_df.iterrows():
        numeric_cols = [col for col in styled_df.columns if col not in excluded_cols]
        numeric_vals = []
        for col in numeric_cols:
            try:
                numeric_vals.append(_to_float(row[col]))
            except Exception:
                numeric_vals.append(float("inf"))

        min_val = min(numeric_vals) if numeric_vals else float("inf")

        for col in numeric_cols:
            try:
                current_val = _to_float(row[col])
                if current_val == min_val and current_val != float("inf"):
                    if isinstance(row[col], str) and row[col].startswith("\\textbf{"):
                        continue
                    styled_df.at[idx, col] = f"\\textbf{{{row[col]}}}"
            except Exception:
                continue

    return styled_df


def dataframe_to_latex(df: pd.DataFrame) -> str:
    """Render a comparison dataframe to LaTeX."""
    return _bold_row_minima(df).to_latex(index=False, escape=False)
