import torch

from dicee.knowledge_graph import KG
from dicee.static_funcs import read_or_load_kg
from dicee.static_funcs import save_checkpoint_model

from src.config import get_default_arguments
from src.static_train_utils import get_literal_dataset, get_model


def test_kbln_initializes_subclass_and_scores():
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
    scores = kge_model.forward_k_vs_all(train_batch)

    assert type(kge_model).__name__ == "Keci_KBLN"
    assert scores.shape[0] == train_batch.shape[0]
    assert scores.shape[1] == entity_dataset.num_entities
    assert kge_model.nf_weights.weight.shape[0] == entity_dataset.num_relations
    assert kge_model.n_num_lit == literal_dataset.num_data_properties


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
    kge_model = get_model(args=args, entity_dataset=entity_dataset)

    checkpoint_path = tmp_path / "kbln_model.pt"
    save_checkpoint_model(kge_model, str(checkpoint_path))
    assert checkpoint_path.exists()
