#!/bin/bash

models=("DistMult" "Keci" "ComplEx" "OMult" "QMult" "DualE" "DeCaL")
datasets=("mutagenesis")
epochs=100

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
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --combined_training --num_epochs $epochs --num_core 20
      else
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --num_epochs $epochs --num_core 20
      fi

      echo
    done
  done
done
