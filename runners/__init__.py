"""
Training runners for KGE and literal embeddings.
"""

def train_kge_model(*args, **kwargs):
    from .kge_runner import train_kge_model as _train_kge_model
    return _train_kge_model(*args, **kwargs)


def train_literals(*args, **kwargs):
    from .literal_runner import train_literals as _train_literals
    return _train_literals(*args, **kwargs)

__all__ = ["train_kge_model", "train_literals"]
