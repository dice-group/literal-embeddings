import torch
import torch.nn as nn

from src.literale import get_literal_dataset


def compute_kbln_statistics(entity_dataset, numerical_literals: torch.Tensor):
    train_triples = torch.as_tensor(entity_dataset.train_set[:, :3], dtype=torch.long)
    head_idx = train_triples[:, 0]
    tail_idx = train_triples[:, 2]
    literal_diff = numerical_literals.index_select(0, head_idx) - numerical_literals.index_select(0, tail_idx)
    c = literal_diff.mean(dim=0)
    var = literal_diff.var(dim=0, unbiased=False) + 1e-6
    return c, var


class KBLNScorer(nn.Module):
    def __init__(self, num_relations: int, numerical_literals: torch.Tensor, c: torch.Tensor, var: torch.Tensor):
        super().__init__()
        self.register_buffer("numerical_literals", numerical_literals.float())
        self.register_buffer("c", c.float())
        self.register_buffer("var", var.float())
        self.num_entities = numerical_literals.size(0)
        self.nf_weights = nn.Embedding(num_relations, numerical_literals.size(1))

    def rbf(self, n: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((n - self.c) ** 2) / self.var)

    def forward(self, head_ids: torch.Tensor, rel_ids: torch.Tensor) -> torch.Tensor:
        head_literals = self.numerical_literals.index_select(0, head_ids.view(-1))
        tail_literals = self.numerical_literals
        literal_diff = head_literals.unsqueeze(1) - tail_literals.unsqueeze(0)
        phi = self.rbf(literal_diff)
        relation_weights = self.nf_weights(rel_ids.view(-1)).unsqueeze(-1)
        return torch.bmm(phi, relation_weights).squeeze(-1)


class KBLNModel(nn.Module):
    def __init__(self, base_model: nn.Module, kbln_scorer: KBLNScorer):
        super().__init__()
        self.base_model = base_model
        self.kbln_scorer = kbln_scorer
        self.args = getattr(base_model, "args", {})
        self.name = getattr(base_model, "name", type(base_model).__name__)

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
        base_scores = self.base_model.forward_k_vs_all(x)
        numeric_scores = self.kbln_scorer(x[:, 0].long(), x[:, 1].long())
        return base_scores + numeric_scores

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_k_vs_all(x)

    @property
    def model(self):
        return self

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base_model = super().__getattr__("base_model")
            return getattr(base_model, name)


def attach_kbln_model(args, kge_model, entity_dataset, literal_dataset=None):
    if literal_dataset is None:
        literal_dataset = get_literal_dataset(args, entity_dataset)
    numerical_literals = literal_dataset.get_literal_value_tensor()
    c, var = compute_kbln_statistics(entity_dataset, numerical_literals)
    kbln_scorer = KBLNScorer(
        num_relations=entity_dataset.num_relations,
        numerical_literals=numerical_literals,
        c=c,
        var=var,
    )
    return KBLNModel(kge_model, kbln_scorer), literal_dataset
