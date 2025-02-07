# literal-embeddings
Neural Regression with Embeddings for Numeric Attribute ( Literals ) Prediction in Knowledge Graphs

## Installation and Requirements
To execute the Literal Embedding model, please install the [dice-embeddings framework](https://github.com/dice-group/dice-embeddings)

##  Command Line Arguments

The following arguments can be used to configure the script. Default values are provided, and they will be used unless explicitly provided as CLI arguments.

### Default Arguments

- `--dataset_dir`: `KGs / FamilyT`
- `--lit_dataset_dir`: `KGs/FamilyL` 
- `--batch_size`: `1024`
- `--num_epochs`: `100`
- `--embedding_dim`: `128`
- `--lr`: `0.05`
- `--optimize_with_literals`: `True`
- `--lit_lr`: `0.0001`
- `--save_embeddings_as_csv`: `False`
- `--save_experiment`: `False`

### `--optimize_with_literals`
- If set to `True`, both the Knowledge Graph Embedding (KGE) model and the Literal Embedding model are trained together in a combined manner. If set to `False` (default), only the KGE model is trained.


### Usage

These arguments need to be passed only if you wish to override the default values. For example:

```bash
python main.py --batch_size 64 --num_epochs 20 --lr 0.0005 
