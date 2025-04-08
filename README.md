# literal-embeddings
Neural Regression with Embeddings for Numeric Attribute ( Literals ) Prediction in Knowledge Graphs

## Installation and Requirements

Create Conda Environment:
```bash
git clone https://github.com/dice-group/literal-embeddings.git
```
```bash
cd literal-embeddings
```

```bash
conda create --name litem  python=3.10 && conda activate litem
```
Install Dice embedding Framework
```bash
pip install -r requirements.txt
```

## Dataset with Literals

The zip file KGs.zip contains all the datasets for experiments. For the first use, 



```bash
 unzip KGs.zip 
```
##  Experiment Types
Currently for the compatability and reproducability of result, KGE models trained with this repo is only supproted to train literal embedding model. To train a KGE model use the following command:

```bash
python main.py  --model Keci --dataset_dir KGs/Family --lr 0.05 --embedding_dim 128 --num_epochs 256 --save_experiment
```
If you want to save the experiment on a desired path, provide the path in the CLI as:
```bash
python main.py  --model Keci --dataset_dir KGs/Family --lr 0.05 --embedding_dim 128 --num_epochs 256 --save_experiment --full_storage_path "Experiments/test_dir"
```
You can then provide such experiment path to train the Literal embedding Model.

There are primarily three different types of experiments available with this framework:

### Training of Literal Embedding Model with Pre-trained KGE model

To train the Literal Embedding model with Pre-trained KGE models, set the flag --literal_training in the args. Use the flags such as --lit_lr and --num_epochs_lit for learning rate and number of epochs for training the Literal Embedding Model. For training with the Liteal Embedding model, provide the path to the folder with weights and configurations of the KGE model using `--pretrained_kge_path`. The literal dataset should be present inside the Literal folder of the respective KGs dataset for the link prediction task.

Important: Use the `--pretrained_kge_path` to provide the path for the pre-trained KGE model with configurations and model weights. The Literal Embedding model takes the Knowledge Graph  and Embedding dim from the pre-trained KGE configs.

```bash
python main.py --dataset_dir KGs/Family --lit_lr 0.05 --literal_training --pretrained_kge_path "Experiments/test_dir" --lit_epochs 200
```
### Training of Knowledge Graph Embedding  with Literal Embedding Model model

The flag combined training must be used to perform combined training of any KGE model and Literal Embedding Model. `--combined_training`. We support all the KGE models available in the dice-embedding framework.

```bash
python main.py  --model Keci --dataset_dir KGs/Family --lr 0.05 --embedding_dim 128 --num_epochs 256 --combined_training
```

### Training of Knowledge Graph Embedding  model


To perform the training of the stand-alone Knowledge Graph Embedding model, just provide the model name, KGs dir and so on. 

```bash
python main.py  --model Keci --dataset_dir KGs/Family --lr 0.05 --embedding_dim 128 --num_epochs 256 
```
