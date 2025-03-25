# literal-embeddings
Neural Regression with Embeddings for Numeric Attribute ( Literals ) Prediction in Knowledge Graphs

## Installation and Requirements
To execute the Literal Embedding model, please install the [dice-embeddings framework](https://github.com/dice-group/dice-embeddings). The requirements from dice-embedding framework is sufficient to run this project.

## Dataset with Literals

The zip file KGs.zip contains all the dataset for experiments. For the first use, 



```bash
 unzip KGs.zip 
```
##  Experiment Types

There are primairly two different types of experiments avaiable with this framework:

### Training of Literal Embedding Model with Pre-traiend KGE model

To train the Literal Embedding model with Pre-trained KGE models, set the flag --literal_training in the args. Use the flags such as --lit_lr and --num_epochs_lit for liearning rate and number of epochs for training the Literal Embedding Model. For training with Liteal Embedding model, provide the path to the folder with weights and configurations of the KGE model using `--pretrained_kge_path`. The literal dataset should be present inside the Literal folder of respective KGs dataset for link prediction task.

Important: Provide the path for pretrained KGE model with configurations and model weights using the `--pretrained_kge_path`. Literal Embedding model takes the Knowledge Graph and Embedding dim from the pre-trained KGE configs.

```bash
python main.py --dataset_dir KGs/Family --lit_lr 0.05 --literal_training --pretrained_kge_path "Experiments/test_dir" --lit_epochs 200
```
### Training of Knowledge Graph Embedding  with Literal Embedding Model model

To perfrom combined training of any KGE model and Literal Embedding Model, the flag combined trainng must be used. `--combined_training`.

```bash
python main.py  --model Keci --dataset_dir KGs/Family --lr 0.05 --embedding_dim 128 --num_epochs 256 --combined_training
```

### Training of Knowledge Graph Embedding  model


To perfrom the training of Knowledge Graph Embedding model, just provide the model name, KGs dir and so on. 

```bash
python main.py  --model Keci --dataset_dir KGs/Family --lr 0.05 --embedding_dim 128 --num_epochs 256 
```