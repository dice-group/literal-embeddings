#!/bin/bash

models=("TransE" "DistMult" "Keci" "ComplEx" "OMult" "QMult" "DeCaL" "Pykeen_MuRE")
#datasets=("Synthetic" "Synthetic_random")
datasets=("Synthetic")
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    storage_path="Experiments/Ablations/${dataset}/${model}"

    echo "=================================="
    echo "Model: $model | Dataset: $dataset"
    echo "Storage Path: $storage_path"
    echo "=================================="

    python main.py \
      --dataset_dir "KGs/$dataset" \
      --model "$model" \
      --combined_training \
      --full_storage_path "$storage_path" \
      --skip_eval_literals 

    echo
  done
done
