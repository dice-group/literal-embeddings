"""
Analysis and evaluation utilities for literal embeddings experiments.

This module contains utilities for:
- Ablation studies
- Baseline calculations
- Performance evaluation
- Statistical analysis
"""

from .ablations import calculate_abalation_scores, evaluate_ablations
from .baselines import evaluate_LOCAL_GLOBAL, calculate_baselines
from .table_latex import dataframe_to_latex
from .table_utils import (
    build_comparison_dataframe,
    build_comparison_dataframe_models,
    load_relation_mappings,
)

__all__ = [
    "calculate_abalation_scores",
    "evaluate_ablations", 
    "evaluate_LOCAL_GLOBAL",
    "calculate_baselines",
    "build_comparison_dataframe",
    "build_comparison_dataframe_models",
    "dataframe_to_latex",
    "load_relation_mappings",
]
