import torch

from dicee.knowledge_graph import KG
from dicee.static_funcs import read_or_load_kg

from src.config import get_default_arguments
from src.static_train_utils import (
    get_literal_dataset,
    get_model,
)


def test_literale_initializes_subclass_and_scores():
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

    assert type(kge_model).__name__ == "Keci_LiteralE"
    assert kge_model.n_num_lit == literal_dataset.num_data_properties

    train_batch = torch.as_tensor(entity_dataset.train_set[:4, :2], dtype=torch.long)
    scores = kge_model.forward_k_vs_all(train_batch)
    assert scores.shape[0] == train_batch.shape[0]
    assert scores.shape[1] == entity_dataset.num_entities
