import torch

from dicee.knowledge_graph import KG
from dicee.static_funcs import read_or_load_kg

from src.config import get_default_arguments
from src.literale import (
    attach_literale_embeddings,
    get_literal_dataset,
)
from src.static_train_utils import (
    get_model,
)


def test_literale_wraps_entity_embeddings_and_scores():
    args = get_default_arguments([])
    args.model = "Keci"
    args.dataset_dir = "KGs/Family"
    args.backend = "pandas"
    args.embedding_dim = 16
    args.scoring_technique = "KvsAll"
    args.literalE = True
    args.eval_model = None

    entity_dataset = read_or_load_kg(args, KG)
    args.num_entities = entity_dataset.num_entities
    args.num_relations = entity_dataset.num_relations
    literal_dataset = get_literal_dataset(args, entity_dataset)
    kge_model = get_model(args=args, entity_dataset=entity_dataset)

    literal_entity_id = literal_dataset.triples[0, 0].item()
    entity_idx = torch.tensor([literal_entity_id], dtype=torch.long)
    base_embedding = kge_model.entity_embeddings(entity_idx).detach().clone()

    attach_literale_embeddings(kge_model, literal_dataset)

    literal_embedding = kge_model.entity_embeddings(entity_idx)
    assert literal_embedding.shape == base_embedding.shape
    assert not torch.allclose(base_embedding, literal_embedding)

    train_batch = torch.as_tensor(entity_dataset.train_set[:4, :2], dtype=torch.long)
    scores = kge_model.forward_k_vs_all(train_batch)
    assert scores.shape[0] == train_batch.shape[0]
    assert scores.shape[1] == entity_dataset.num_entities
