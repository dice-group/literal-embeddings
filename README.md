# literal-embeddings
Neural Regression with Embeddings for Numeric Attribute ( Literals ) Prediction in Knowledge Graphs

## Installation and Requirements
To execute the Literal Embedding model, please install the [dice-embeddings framework](https://github.com/dice-group/dice-embeddings)

##  Configurable Arguments

The following arguments can be used to configure the script. Default values are provided, and they will be used unless explicitly modified on the main script.

### Default Arguments

- `--dataset_dir`: `KGs / FamilyT`
- `--dataset_dir_lit`: `KGs/FamilyL` 
- `--batch_size`: `1024`
- `--num_epochs`: `100`
- `--embedding_dim`: `128`
- `--lr`: `0.05`
- `--combined_training`: `True`
- `--lit_lr`: `0.0001`
- `--lit_epochs`: `500`
- `--save_embeddings_as_csv`: `False`
- `--save_experiment`: `False`
- `--pretrained_kge`: `False`
- `--pretrained_kge_path`: `None`

### `--combined_training`
- If set to `True`, both the Knowledge Graph Embedding (KGE) model and the Literal Embedding model are trained together in a combined manner. If set to `False` (default), only the KGE model is trained.

### `--pretrained_kge`
If pretrained_kge is set to true, provide a valid path to a folder that contains a pre-trained KGE model. This will bypass all the combined training procedures and only run an instance of LiteralEmbedding model.


### Usage

Set the desired args within the main script and run 
```bash
python main.py
