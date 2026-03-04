import torch
import torch.nn as nn

from src.dataset import LiteralDataset


class Gate(nn.Module):
    def __init__(self, input_size, output_size, gate_activation=torch.sigmoid):
        super().__init__()
        literal_size = input_size - output_size
        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(literal_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], dim=-1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        return (1 - gate) * x_ent + gate * g_embedded


class LiteralEEntityEmbedding(nn.Module):
    """Drop-in entity embedding wrapper that fuses numerical literals."""

    def __init__(self, base_embedding: nn.Embedding, numerical_literals: torch.Tensor):
        super().__init__()
        if not hasattr(base_embedding, "weight"):
            raise TypeError("LiteralE requires an embedding layer with a weight parameter.")
        self.base_embedding = base_embedding
        self.embedding_dim = base_embedding.embedding_dim
        self.num_embeddings = base_embedding.num_embeddings
        self.padding_idx = getattr(base_embedding, "padding_idx", None)
        self.max_norm = getattr(base_embedding, "max_norm", None)
        self.norm_type = getattr(base_embedding, "norm_type", 2.0)
        self.scale_grad_by_freq = getattr(base_embedding, "scale_grad_by_freq", False)
        self.sparse = getattr(base_embedding, "sparse", False)
        self.register_buffer("numerical_literals", numerical_literals.float())
        self.emb_num_lit = Gate(self.embedding_dim + self.numerical_literals.size(1), self.embedding_dim)
        self._cached_weight = None

    def _compose(self, entity_embeddings: torch.Tensor, literals: torch.Tensor) -> torch.Tensor:
        return self.emb_num_lit(entity_embeddings, literals)

    def prepare_batch(self, entity_ids: torch.Tensor = None):
        del entity_ids
        self._cached_weight = self._compose(self.base_embedding.weight, self.numerical_literals)

    def clear_batch_cache(self):
        self._cached_weight = None

    def forward(self, entity_idx: torch.Tensor) -> torch.Tensor:
        if self._cached_weight is not None:
            flat_idx = entity_idx.reshape(-1)
            return self._cached_weight.index_select(0, flat_idx).view(*entity_idx.shape, -1)
        base_embeddings = self.base_embedding(entity_idx)
        literals = self.numerical_literals.index_select(0, entity_idx.reshape(-1)).view(*entity_idx.shape, -1)
        return self._compose(
            base_embeddings.reshape(-1, self.embedding_dim),
            literals.reshape(-1, literals.size(-1)),
        ).view(*entity_idx.shape, -1)

    @property
    def weight(self):
        if self._cached_weight is not None:
            return self._cached_weight
        return self._compose(self.base_embedding.weight, self.numerical_literals)


def get_literal_dataset(args, entity_dataset):
    literal_dataset = LiteralDataset(
        dataset_dir=args.dataset_dir,
        ent_idx=entity_dataset.entity_to_idx,
        normalization="min-max",
    )
    args.num_attributes = literal_dataset.num_data_properties
    return literal_dataset


def attach_literale_embeddings(kge_model, literal_dataset):
    if not hasattr(kge_model, "entity_embeddings"):
        raise ValueError("LiteralE requires the KGE model to expose `entity_embeddings`.")
    if isinstance(kge_model.entity_embeddings, LiteralEEntityEmbedding):
        return kge_model
    kge_model.entity_embeddings = LiteralEEntityEmbedding(
        base_embedding=kge_model.entity_embeddings,
        numerical_literals=literal_dataset.get_literal_value_tensor(),
    )
    return kge_model
