#!/bin/bash

models=("ComplEx" "OMult" "QMult")
datasets=("YAGO15k" "DB15K")
epochs=300

for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    for mode in "combined" "separate" "literalE" "kbln"
    do
      echo "=================================="
      echo "Model: $model | Dataset: $dataset | Mode: $mode"
      echo "=================================="

      if [ "$mode" == "combined" ]; then
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --combined_training --num_epochs $epochs --embedding_dim 64  --num_core 20 
      elif [ "$mode" == "literalE" ]; then
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --literalE --num_epochs $epochs --embedding_dim 64 --num_core 20
      elif [ "$mode" == "kbln" ]; then
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --kbln --num_epochs $epochs --embedding_dim 64 --num_core 20
      else
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --num_epochs $epochs --embedding_dim 64 --num_core 20
      fi

      echo
    done
  done
done
