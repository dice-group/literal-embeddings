"""Training runners for the KGE-only codebase."""

from .kge_runner import train_kge_model
from .runner_KGEntText import train_kgenttext_model

__all__ = [
    "train_kge_model",
    "train_kgenttext_model",
]
