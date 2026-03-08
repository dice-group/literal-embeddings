#!/bin/bash

models=("ComplEx" "OMult" "Keci" "QMult" "DistMult" "Pykeen_RotatE")
datasets=("YAGO15k" "FB15k-237")


for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    exp_dir="Experiments/KGE_LitEm_all_triples/${dataset}-${model}-128"

    echo "==================================================="
    echo "Model: $model | Dataset: $dataset"
    echo "Using pretrained path: $exp_dir"
    echo "==================================================="

    python main.py \
      --dataset_dir "KGs/$dataset" \
      --model "$model" \
      --literal_training \
      --num_literal_runs 5 \
      --pretrained_kge_path "$exp_dir" \
      --use_best_config


    echo
  done
done