#!/bin/bash

models=("TransE" "DistMult" "Keci" "ComplEx" "OMult" "QMult" "DeCaL" "Pykeen_MuRE")
datasets=("FB15k-237" "DB15K" "YAGO15k" "mutagenesis")


for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    exp_dir="Experiments/KGE/${dataset}/${model}"

    echo "==================================================="
    echo "Model: $model | Dataset: $dataset"
    echo "Using pretrained path: $exp_dir"
    echo "==================================================="

    python main.py \
      --dataset_dir "KGs/$dataset" \
      --model "$model" \
      --literal_training \
      --num_literal_runs 3 \
      --pretrained_kge_path "$exp_dir"

    echo
  done
done