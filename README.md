# literal-embeddings

This repository is trimmed to the training pipeline for:

- DICEE knowledge graph embedding models
- Clifford-based KGE models in `src/clifford.py`

## Setup

```bash
conda create --name litem python=3.10
conda activate litem
pip install -r requirements.txt
```

Unpack the bundled datasets if needed:

```bash
unzip KGs.zip
```

Each dataset should look like:

```text
KGs/<dataset>/
  train.txt
  valid.txt
  test.txt
```

## Training

Train a standalone DICEE KGE model:

```bash
python main.py \
  --model Keci \
  --dataset_dir KGs/Family \
  --embedding_dim 128 \
  --num_epochs 256 \
  --lr 0.05
```

## Notes

- `main.py` dispatches directly to KGE training.
- `runners/kge_runner.py` handles the DICEE and Clifford KGE training path.
- `src/clifford.py` contains the custom Clifford KGE architectures.
- The remaining test covers the Clifford KGE path.
