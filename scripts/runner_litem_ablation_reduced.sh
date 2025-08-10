#!/bin/bash

model="TransE"
datasets=("YAGO15k" "FB15k-237")
embedding_dim=100
for dataset in "${datasets[@]}"; do
  exp_dir="Experiments/KGE_LitEm_all_triples/${dataset}-${model}-${embedding_dim}"

  for ratio in 100 80 60 40 20; do
    sampling_ratio=$(echo "$ratio / 100" | bc -l)
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
      --model "$model"

    echo
  done
done
