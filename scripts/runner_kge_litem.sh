#!/bin/bash

exp_models=("TransE")  # Make sure the model name casing is correct
dataset_names=("FB15k-237" "DB15K" "YAGO15k" "mutagenesis")

embedding_dim=100
num_epochs=300

for model in "${exp_models[@]}"; do
  for dataset in "${dataset_names[@]}"; do

    path_single_kg="KGs/${dataset}/${dataset}_EntityTriples.txt"
    full_storage_path="Experiments/KGE_LitEm_all_triples/${dataset}-${model}-${embedding_dim}"

    echo "==========================================================="
    echo "Model: $model | Dataset: $dataset"
    echo "Single KG Path: $path_single_kg"
    echo "Storage Path: $full_storage_path"
    echo "==========================================================="

    python main.py \
      --model "$model" \
      --embedding_dim $embedding_dim \
      --num_epochs $num_epochs \
      --train_all_triples \
      --path_single_kg "$path_single_kg" \
      --full_storage_path "$full_storage_path"

    echo
  done
done
