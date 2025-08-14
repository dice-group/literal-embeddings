#!/bin/bash

models=("DistMult" "Keci" "ComplEx" "OMult" "QMult" "DualE" "DeCaL")
datasets=("FB15k-237" "YAGO15k" "DB15K")
epochs=200

for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    for mode in "combined" "separate"
    do
      echo "=================================="
      echo "Model: $model | Dataset: $dataset | Mode: $mode"
      echo "=================================="

      if [ "$mode" == "combined" ]; then
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --combined_training --num_epochs $epochs --embedding_dim 64 
      else
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --num_epochs $epochs --embedding_dim 64
      fi

      echo
    done
  done
done
