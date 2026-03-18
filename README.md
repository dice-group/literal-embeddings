# KGE Training Pipeline

This repository now keeps a reduced scope:

- train a DICE-compatible knowledge graph embedding model
- save the trained experiment artifacts
- load a stored experiment directly with DICE `KGE(path=...)`

## Train a KGE model

```bash
python main.py --model Keci --dataset_dir KGs/Family --embedding_dim 128 --lr 0.05 --num_epochs 256
```

Stored runs are written under `Experiments/KGE/<dataset>/<model>_<embedding_dim>` unless `--full_storage_path` is provided.

## Load a stored model as DICE `KGE`

```bash
python -c "from dicee.knowledge_graph_embeddings import KGE; model = KGE(path='Experiments/KGE/Family/Keci_128'); print(model.model.name)"
```

This loads the saved model directly through the DICE `KGE` interface.

## Retained code layout

- `main.py`: KGE training entry point
- `runners/kge_runner.py`: training orchestration
- `src/`: minimal training, config, callback, dataset, and storage helpers
