"""Utility helpers for the KGEntText pipeline."""

import pandas as pd
from datasets import load_dataset
from dicee.knowledge_graph_embeddings import KGE


def load_pretrained_kge_components(pretrained_kge_path):
    """Load entity mapping and entity embeddings from a stored KGE experiment."""
    model = KGE(path=pretrained_kge_path)
    entity_to_idx = pd.read_csv(
        f"{pretrained_kge_path}/entity_to_idx.csv",
        index_col=0,
    )
    entity_to_idx = {
        name: idx for idx, name in enumerate(entity_to_idx["entity"].tolist())
    }
    entities = list(entity_to_idx.keys())
    entity_embeddings = model.get_entity_embeddings(entities)
    return {
        "entity_to_idx": entity_to_idx,
        "entity_embeddings": entity_embeddings,
    }


def load_fb15k_entity_name_mapping():
    """Load Freebase entity labels for FB15k-style ids via Hugging Face datasets."""
    dataset = load_dataset(
        "kdm-daiict/freebase-wikidata-mapping",
        split="train",
    )
    mapping = {}
    for row in dataset:
        freebase_id = row.get("freebase_id")
        label = row.get("label")
        if freebase_id and label and freebase_id not in mapping:
            mapping[freebase_id] = label
    return mapping
