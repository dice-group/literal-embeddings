"""
Training runners for KGE and literal embeddings.
"""

from .kge_runner import train_kge_model
from .literal_runner import train_literals

__all__ = ["train_kge_model", "train_literals"]
