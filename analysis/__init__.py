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

__all__ = [
    "calculate_abalation_scores",
    "evaluate_ablations", 
    "evaluate_LOCAL_GLOBAL",
    "calculate_baselines"
]
