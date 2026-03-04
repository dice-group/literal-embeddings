#!/bin/bash

models=("ComplEx" "OMult")
datasets=("YAGO15k")


for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    exp_dir="Experiments/KGE/${dataset}/${model}_64"

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