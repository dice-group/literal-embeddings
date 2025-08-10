#!/bin/bash

model="TransE"
datasets=("FB15k-237" "DB15K" "YAGO15k" "mutagenesis")
exp_dir="Experiments/KGE_LitEm_all_triples"

for dataset in "${datasets[@]}"; do
  pretrained_path="${exp_dir}/${dataset}-${model}-100"
  
  echo "=================================================="
  echo "Model: $model | Dataset: $dataset"
  echo "Pretrained Path: $pretrained_path"
  echo "=================================================="
  echo "LitEM training"
  echo "=================================================="
  python main.py \
    --model "$model" \
    --dataset_dir "KGs/$dataset" \
    --literal_training \
    --num_literal_runs 5 \
    --pretrained_kge_path "$pretrained_path"

  echo "=================================================="
  echo "LitEM training completed"
  echo "==================================================" 

  echo "LitEM training - embeddings- updated"
  echo "=================================================="
  python main.py \
    --model "$model" \
    --dataset_dir "KGs/$dataset" \
    --literal_training \
    --num_literal_runs 5 \
    --pretrained_kge_path "$pretrained_path" \
    --update_entity_embeddings

  echo "=================================================="
  echo "LitEM training completed"
  echo "=================================================="

  if [[ "$dataset" == "FB15k-237" || "$dataset" == "YAGO15k" ]]; then
    echo "=================================================="
    python main.py \
    --model "$model" \
    --dataset_dir "KGs/${dataset}_disjoint" \
    --literal_training \
    --num_literal_runs 5 \
    --pretrained_kge_path "$pretrained_path"
  fi
done
