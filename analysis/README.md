# Analysis Module

This directory contains utilities for analyzing and evaluating literal embeddings experiments.

## Files

- **`ablations.py`** - Functions for running ablation studies and analysis
  - `calculate_abalation_scores()` - Calculate ablation scores for experiments
  - `evaluate_ablations()` - Evaluate different ablation configurations

- **`baselines.py`** - Baseline calculation and comparison utilities
  - `evaluate_LOCAL_GLOBAL()` - Evaluate local and global average predictions
  - `calculate_baselines()` - Calculate baseline metrics for comparison

- **`tables_paper.ipynb`** - Jupyter notebook for generating paper tables and visualizations

- **`Appendix_Neural_Regression.pdf`** - Research document with neural regression appendix

## Usage

```python
from analysis.ablations import calculate_abalation_scores, evaluate_ablations
from analysis.baselines import evaluate_LOCAL_GLOBAL, calculate_baselines

# Run ablation analysis
scores = calculate_abalation_scores("path/to/experiment")

# Calculate baselines
baselines = calculate_baselines("dataset_name")
```
