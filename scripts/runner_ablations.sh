#!/bin/bash

models=("DistMult" "Keci" "ComplEx" "OMult" "QMult" "DeCaL" "DualE" )
datasets=("Synthetic" "Synthetic_random")
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for mode in "combined" "literalE" "kbln"; do
      storage_path="Experiments/Ablations/${dataset}/${model}/${mode}"

      echo "=================================="
      echo "Model: $model | Dataset: $dataset | Mode: $mode"
      echo "Storage Path: $storage_path"
      echo "=================================="

      common_args=(
        --dataset_dir "KGs/$dataset"
        --model "$model"
        --full_storage_path "$storage_path"
        --num_core 20
        --embedding_dim 64
        --num_epochs 300
      )

      if [ "$mode" == "combined" ]; then
        python main.py "${common_args[@]}" --combined_training --skip_eval_literals
      elif [ "$mode" == "literalE" ]; then
        python main.py "${common_args[@]}" --literalE
      else
        python main.py "${common_args[@]}" --kbln
      fi

      echo
    done
  done
done
