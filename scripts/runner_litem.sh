#!/bin/bash

model="TransE"
datasets=("FB15k-237" "DB15K" "YAGO15k" "mutagenesis")
exp_dir="Experiments/KGE_LitEm_all_triples"

for dataset in "${datasets[@]}"; do
  pretrained_path="${exp_dir}/${dataset}_256_100/${model}"
  
  echo "=================================================="
  echo "Model: $model | Dataset: $dataset"
  echo "Pretrained Path: $pretrained_path"
  echo "=================================================="

  python main.py \
    --model "$model" \
    --dataset_dir "KGs/$dataset" \
    --literal_training \
    --num_literal_runs 5 \
    --pretrained_kge_path "$pretrained_path"

  echo
done
