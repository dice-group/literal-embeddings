# literal-embeddings
Official Reposiotry for the paper: Neural Regression with Embeddings for Numeric Attribute ( Literals ) Prediction in Knowledge Graphs

### Supplementry Materials for the paper 
The file `Appendix_Neural_Regression.pdf` contains all the Appendix contents for the paper.

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



There are primarily three different types of experiments available with this framework:

### Training of Literal Embedding Model with Pre-trained KGE model

To train the Literal Embedding model with Pre-trained KGE models, set the flag --literal_training in the args. Use the flags such as --lit_lr and --num_epochs_lit for learning rate and number of epochs for training the Literal Embedding Model. For training with the Liteal Embedding model, provide the path to the folder with weights and configurations of the KGE model using `--pretrained_kge_path`.

This should support KGE model trained with this framework as well as from dice-embedding framework.

 The literal dataset should be present inside the Literal folder of the respective KGs dataset for the link prediction task. e.g, `KGs/Family/literals` should contain numeric triples for Family dataset.

Important: Use the `--pretrained_kge_path` to provide the path for the pre-trained KGE model with configurations and model weights. 
Provide the dataset dir using `--dataset_dir` argument. Keep the literal dataset within the dataset dir inside a folder named "literals"

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
If you wish to save the KGE model for Literal embedding training ( or other purposes), use save_experiment argument

```bash
python main.py  --model Keci --dataset_dir KGs/Family --lr 0.05 --embedding_dim 128 --num_epochs 256 --save_experiment
```

If you want to save the experiment on a desired path, provide the path in the CLI as:
```bash
python main.py  --model Keci --dataset_dir KGs/Family --lr 0.05 --embedding_dim 128 --num_epochs 256 --save_experiment --full_storage_path "Experiments/test_dir"
```
You can then provide such experiment path to train the Literal embedding Model.

## Recreating the results from the paper

### 1. Train the embedding model for LitEm
We use the TransE embedding at d=100, for 256 epochs as the base results with all the triples. To recreate the models, firsh make the scripts runnable and run the script:

```bash
chmod +x scripts/* 
```
```bash
./scripts/runner_kge_litem.sh
```
This will train the TransE Embeddings for all the datasets used in experiments for 256 epochs , d = 100, using KvsAll Scoring Technique

### 2. Recreate the litEm results
```bash
./scripts/runner_litem.sh
```
This will reproduce the results for the LitEm model presented for Numerical Attribute Predictions for all datasets.

### 3. Training the Link Prediction task
```bash
./scripts/runner_link_prediction.sh
```
This will train all the models for Link Prediction task with and without combined training of the LitEm model

### 5. Ablations on reduced 
```bash
./scripts/runner_litem_ablation_reduced.sh
```
This will run the ablation for LitEM model at different sparsity of KGs as mentioned in the paper

### 6. Ablations
```bash
./scripts/runner_litem_ablation.sh
```
To check the ablation of "Literal Awareness" on Synthetic and Synthetic Random datasets, run the script. After this calculate the ablation scores using:
```python
python ablations.py
```
### 7. LitEm experiments on All KGE models and Dataset
To check the Literal Prediction perfromance of the KGE models, run the command:
```bash
./scripts/runner_kge_litpred.sh
```

### 8. Calculate baselines
To calcuate LOCAL and GLOBAL baselines for the knowledge graphs, run the python command:
```python
python calcuate_baselines.py
```
If you download the pre-trained models, you can direclly unzip the contents at Experiments/ and use the notebook tables_paper.ipynb to create the pandas table and also in latex format for the tables present in the paper as well as supplementary materials.

The link for the Experiments:

These scripts are to recreate the experiment results, feel free to try other features of of Literal Embedding framework.

### Extention for other KGs
If you want to extend this to new KGs, create a new folder under KGs/ . Keep all the splits `train.txt`, `test.txt` and `valid.txt`. Create a file containing all the triples at `{dataset_name}_EntityTriples.txt`. For the Literals, keep the literal data inside the `literals` dir inside the dataset folder `KGs/{dataset_name}/literals`. Provide also `train.txt`, `test.txt`, `valid.txt` within the literals folder. Keep all the KGs in triples format with tab sepration. Following this, you can use our approach with any KGs for literal prediction and Augmentation.   

## Using Pre-trained Experiments
If you wish to use the pre-trained weights, download the  zipped experiment files at https://zenodo.org/records/15423527 and Keep the contents in the base directory. The contents should be aligned as `Experiments/KGE`and so on. You can do that by using the `unzip Experiments.zip -d .`.
If you download the pre-trained models, you do not need to run any experiments. Use `tables_paper.ipynb`to re-create results from the experiment files.