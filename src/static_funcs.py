import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import torch
from dicee.knowledge_graph_embeddings import KGE
from dicee.static_funcs import intialize_model


@dataclass
class KGEModelComponents:
    model: Any
    config: Dict
    entity_to_idx: Dict[str, int]
    relation_to_idx: Dict[str, int]
    model_path: Optional[str] = None

    @property
    def embedding_dim(self) -> int:
        return self.model.embedding_dim

    @property
    def model_name(self) -> str:
        return self.model.name

    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_entities": len(self.entity_to_idx),
            "num_relations": len(self.relation_to_idx),
            "model_path": self.model_path,
        }


def save_kge_experiments(args, loss_log=None):
    if args is None:
        raise ValueError("`args` must be provided to save experiment results.")

    os.makedirs(args.full_storage_path, exist_ok=True)

    if loss_log is not None:
        df_loss_log = pd.DataFrame.from_dict(loss_log, orient="index").transpose()
        loss_log_path = os.path.join(args.full_storage_path, "loss_log.tsv")
        df_loss_log.to_csv(loss_log_path, sep="\t", index=False)

    args.device = str(args.device)
    config_path = os.path.join(args.full_storage_path, "configuration.json")
    with open(config_path, "w") as file:
        json.dump(vars(args), file, indent=4)


def get_full_storage_path(args):
    if args.full_storage_path:
        return args.full_storage_path

    dataset_name = None
    if getattr(args, "dataset_dir", None):
        dataset_name = os.path.basename(args.dataset_dir.rstrip("/"))
    elif getattr(args, "path_single_kg", None):
        dataset_name = os.path.basename(args.path_single_kg.rstrip("/"))
    if not dataset_name:
        dataset_name = "unknown_dataset"

    if getattr(args, "test_runs", False):
        exp_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
        return f"Experiments/Test_runs/{exp_time}"

    model_name = getattr(args, "model", "unknown_model")
    embedding_dim = getattr(args, "embedding_dim", "unknown_dim")
    return f"Experiments/KGE/{dataset_name}/{model_name}_{embedding_dim}"


def load_model_components(kge_path: str) -> Optional[KGEModelComponents]:
    kge_model = None
    config = None
    entity_to_idx = None
    relation_to_idx = None

    try:
        kge_obj = KGE(path=kge_path)
        kge_model = kge_obj.model
        config = kge_obj.configs
        entity_to_idx = kge_obj.entity_to_idx
        relation_to_idx = kge_obj.relation_to_idx
        print("DICE KGE model loaded successfully.")
    except Exception:
        try:
            config_path = os.path.join(kge_path, "configuration.json")
            model_path = os.path.join(kge_path, "model.pt")
            entity_to_idx_path = os.path.join(kge_path, "entity_to_idx.csv")
            relation_to_idx_path = os.path.join(kge_path, "relation_to_idx.csv")

            if not all(
                os.path.exists(path)
                for path in [
                    config_path,
                    model_path,
                    entity_to_idx_path,
                    relation_to_idx_path,
                ]
            ):
                raise FileNotFoundError("Required KGE artifacts are missing.")

            with open(config_path, "r") as file:
                config = json.load(file)

            entity_to_idx_df = pd.read_csv(entity_to_idx_path)
            relation_to_idx_df = pd.read_csv(relation_to_idx_path)
            entity_to_idx = {
                row["entity"]: idx for idx, row in entity_to_idx_df.iterrows()
            }
            relation_to_idx = {
                row["relation"]: idx for idx, row in relation_to_idx_df.iterrows()
            }

            weights = torch.load(model_path, map_location="cpu")
            kge_model, _ = intialize_model(config, 0)
            kge_model.load_state_dict(weights)
            print("Manual KGE load successful.")
        except Exception as error:
            print("Building the KGE model failed.", str(error))
            return None

    return KGEModelComponents(
        model=kge_model,
        config=config,
        entity_to_idx=entity_to_idx,
        relation_to_idx=relation_to_idx,
        model_path=kge_path,
    )


def create_and_save_report(trainer):
    num_entities = trainer.kge_model.num_entities
    num_relations = trainer.kge_model.num_relations
    train_triples = len(trainer.trainer.entity_dataset.train_set)
    total_params = sum(parameter.numel() for parameter in trainer.kge_model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)
    path_experiment = trainer.args.full_storage_path
    run_time = datetime.now() - trainer.start_time
    runtime_seconds = run_time.total_seconds()
    print("Total runtime of the experiment", runtime_seconds)

    report = {
        "num_train_triples": train_triples,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "max_length_subword_tokens": None,
        "runtime_kg_loading": getattr(trainer.args, "_kg_loading_time", None),
        "EstimatedSizeMB": model_size_mb,
        "NumParam": total_params,
        "path_experiment_folder": path_experiment,
        "Runtime": runtime_seconds,
    }
    report_path = f"{path_experiment}/report.json"
    with open(report_path, "w") as file:
        json.dump(report, file, indent=4)
