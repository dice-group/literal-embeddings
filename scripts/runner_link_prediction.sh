#!/bin/bash

models=("DistMult" "Keci" "ComplEx" "OMult" "QMult" "DualE" "DeCaL")
datasets=("FB15k-237" "YAGO15k" "DB15K" "mutagenesis")
epochs=500

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
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --combined_training --num_epochs $epochs --num_core 20 --eval_every_n_epochs 50
      else
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --num_epochs $epochs --num_core 20 --eval_every_n_epochs 50
      fi

      echo
    done
  done
done
