import pytest
import torch

from src.config import get_default_arguments


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_clnn_forward_forward_smoke():
    from src.kge_models import CLNN

    args = get_default_arguments([])
    args.model = "CLNN"
    args.scoring_technique = "NegSample"
    args.embedding_dim = 32
    args.lr = 1e-2
    args.learning_rate = args.lr
    args.num_entities = 64
    args.num_relations = 16

    model = CLNN(vars(args))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    batch_size = 8
    x_pos = torch.empty(batch_size, 3, dtype=torch.long)
    x_neg = torch.empty(batch_size, 3, dtype=torch.long)

    x_pos[:, 0] = torch.randint(0, args.num_entities, (batch_size,))
    x_pos[:, 1] = torch.randint(0, args.num_relations, (batch_size,))
    x_pos[:, 2] = torch.randint(0, args.num_entities, (batch_size,))

    x_neg.copy_(x_pos)
    x_neg[:, 2] = torch.randint(0, args.num_entities, (batch_size,))

    report = model.ff_update(x_pos, x_neg, optimizer)
    assert "loss" in report
    assert torch.isfinite(torch.tensor(report["loss"], dtype=torch.float32))

    h, r, t = model.get_triple_representation(x_pos)
    score = model.score(h, r, t)
    assert score.shape == (batch_size,)
    assert torch.isfinite(score).all()


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_clnn_forward_forward_smoke_g0():
    from src.kge_models import CLNN

    args = get_default_arguments([])
    args.model = "CLNN"
    args.scoring_technique = "NegSample"
    args.embedding_dim = 32
    args.lr = 1e-2
    args.learning_rate = args.lr
    args.num_entities = 64
    args.num_relations = 16
    args.clifford_g = []

    model = CLNN(vars(args))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    batch_size = 8
    x_pos = torch.empty(batch_size, 3, dtype=torch.long)
    x_neg = torch.empty(batch_size, 3, dtype=torch.long)

    x_pos[:, 0] = torch.randint(0, args.num_entities, (batch_size,))
    x_pos[:, 1] = torch.randint(0, args.num_relations, (batch_size,))
    x_pos[:, 2] = torch.randint(0, args.num_entities, (batch_size,))

    x_neg.copy_(x_pos)
    x_neg[:, 2] = torch.randint(0, args.num_entities, (batch_size,))

    report = model.ff_update(x_pos, x_neg, optimizer)
    assert "loss" in report
    assert torch.isfinite(torch.tensor(report["loss"], dtype=torch.float32))

    h, r, t = model.get_triple_representation(x_pos)
    score = model.score(h, r, t)
    assert score.shape == (batch_size,)
    assert torch.isfinite(score).all()
