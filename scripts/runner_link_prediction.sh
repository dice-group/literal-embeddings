#!/bin/bash

models=("TransE" "DistMult" "Keci" "ComplEx" "OMult" "QMult" "DeCaL" "Pykeen_MuRE")
datasets=("FB15k-237" "DB15K" "YAGO15k" "mutagenesis")
epochs=256

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
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --combined_training --num_epochs $epochs 
      else
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --num_epochs $epochs
      fi

      echo
    done
  done
done
