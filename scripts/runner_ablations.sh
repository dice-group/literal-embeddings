#!/bin/bash

models=("DistMult" "Keci" "ComplEx" "OMult" "QMult" "DeCaL" "DualE" )
datasets=("Synthetic" "Synthetic_random")
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
      --skip_eval_literals \
      --num_core 20 \
      --embedding_dim 64 \
      --num_epochs 200

    echo
  done
done
