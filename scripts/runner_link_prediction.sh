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
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --combined_training --num_epochs $epochs --embedding_dim 64 --literal_model "clifford" --num_core 20 --scoring_technique "1vsAll" --lr 0.05  --batch_size 512 --eval_every_n_epochs 50
      else
        python main.py --dataset_dir "KGs/$dataset" --model "$model" --num_epochs $epochs --embedding_dim 64 --num_core 20 --scoring_technique "1vsAll" --lr 0.05 --batch_size 512 --eval_every_n_epochs 50 
      fi

      echo
    done
  done
done
