#!/bin/bash

model="TransE"
datasets=("YAGO15k" "FB15k-237")
embedding_dim=128
for dataset in "${datasets[@]}"; do
  exp_dir="Experiments/KGE_LitEm_all_triples/${dataset}-${model}-${embedding_dim}"

  for ratio in 80 60 40 20; do
    sampling_ratio="0.$ratio"
    storage_path="Experiments/Ablations/${dataset}_${ratio}/${model}"

    echo "===================================================="
    echo "Dataset: $dataset | Model: $model | Ratio: $sampling_ratio"
    echo "Pretrained Path: $exp_dir"
    echo "Storage Path: $storage_path"
    echo "===================================================="

    python main.py \
      --dataset_dir "KGs/$dataset" \
      --lit_sampling_ratio "$sampling_ratio" \
      --pretrained_kge_path "$exp_dir" \
      --literal_training \
      --full_storage_path "$storage_path" \
      --model "$model" \
      --use_best_config \
      --num_literal_runs 3

    echo
  done
done
