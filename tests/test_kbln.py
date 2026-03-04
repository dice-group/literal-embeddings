import torch

from dicee.knowledge_graph import KG
from dicee.static_funcs import read_or_load_kg
from dicee.static_funcs import save_checkpoint_model

from src.config import get_default_arguments
from src.kbln import attach_kbln_model
from src.literale import get_literal_dataset
from src.static_train_utils import get_model


def test_kbln_wraps_kge_and_augments_scores():
    args = get_default_arguments([])
    args.model = "Keci"
    args.dataset_dir = "KGs/Family"
    args.backend = "pandas"
    args.embedding_dim = 16
    args.scoring_technique = "KvsAll"
    args.kbln = True
    args.eval_model = None

    entity_dataset = read_or_load_kg(args, KG)
    args.num_entities = entity_dataset.num_entities
    args.num_relations = entity_dataset.num_relations
    literal_dataset = get_literal_dataset(args, entity_dataset)
    kge_model = get_model(args=args, entity_dataset=entity_dataset)

    train_batch = torch.as_tensor(entity_dataset.train_set[:4, :2], dtype=torch.long)
    base_scores = kge_model.forward_k_vs_all(train_batch).detach().clone()

    kbln_model, _ = attach_kbln_model(args, kge_model, entity_dataset, literal_dataset=literal_dataset)
    wrapped_scores = kbln_model.forward_k_vs_all(train_batch)

    assert wrapped_scores.shape == base_scores.shape
    assert not torch.allclose(base_scores, wrapped_scores)
    assert kbln_model.kbln_scorer.nf_weights.weight.shape[0] == entity_dataset.num_relations


def test_kbln_checkpoint_save(tmp_path):
    args = get_default_arguments([])
    args.model = "Keci"
    args.dataset_dir = "KGs/Family"
    args.backend = "pandas"
    args.embedding_dim = 16
    args.scoring_technique = "KvsAll"
    args.kbln = True
    args.eval_model = None

    entity_dataset = read_or_load_kg(args, KG)
    args.num_entities = entity_dataset.num_entities
    args.num_relations = entity_dataset.num_relations
    literal_dataset = get_literal_dataset(args, entity_dataset)
    kge_model = get_model(args=args, entity_dataset=entity_dataset)
    kbln_model, _ = attach_kbln_model(args, kge_model, entity_dataset, literal_dataset=literal_dataset)

    checkpoint_path = tmp_path / "kbln_model.pt"
    save_checkpoint_model(kbln_model, str(checkpoint_path))
    assert checkpoint_path.exists()
