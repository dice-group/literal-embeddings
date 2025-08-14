#!/bin/bash

# Models to test (base models and their literal variants)
base_models=("complex" "distmult" "conve" "tucker")
datasets=("FB15K-237" "YAGO15K")

# Training configuration (optimal parameters from experimental setup)
embedding_dim=64
batch_size=256
lr=0.001
num_iterations=300

echo "=========================================="
echo "Running Base Model vs Combined Training vs Standalone Literal Experiments"
echo "=========================================="

for base_model in "${base_models[@]}"; do
  for dataset in "${datasets[@]}"; do

    echo "==========================================================================================================="
    echo "Testing model: $base_model | dataset: $dataset"
    echo "==========================================================================================================="
    
    # Create base experiment directory
    base_exp_dir="Experiments/${base_model}_${dataset}"
    mkdir -p "$base_exp_dir"
    
    # 1. Run base model only (no literal information)
    echo ">>> Running BASE MODEL ONLY (no literals): $base_model"
    
    cmd_base="python main.py \
      --dataset \"$dataset\" \
      --model \"$base_model\" \
      --batch_size $batch_size \
      --num_iterations $num_iterations \
      --lr $lr \
      --embedding_dim $embedding_dim \
      --eval_at_epochs 200 250 256 \
      --exp_dir \"$base_exp_dir\"" 
      
      
    
    echo "Command: $cmd_base"
    eval $cmd_base
    echo "✓ Completed BASE MODEL: $base_model | $dataset"
    echo "-----------------------------------------------------------------------------------------------------------"
    
    # 2. Run combined training with base model + literal embeddings  
    echo ">>> Running COMBINED TRAINING: $base_model + literal embeddings"
    
    exp_dir_combined="${base_exp_dir}/LitEM"
    mkdir -p "$exp_dir_combined"
    
    cmd_combined="python main.py \
      --dataset \"$dataset\" \
      --model \"$base_model\" \
      --batch_size $batch_size \
      --num_iterations $num_iterations \
      --lr $lr \
      --embedding_dim $embedding_dim \
      --combined_training \
      --dynamic_weighting \
      --eval_at_epochs 200 250 256 \
      --exp_dir \"$exp_dir_combined\"" 
      
    
    echo "Command: $cmd_combined"
    eval $cmd_combined
    echo "✓ Completed COMBINED TRAINING: $base_model + literal embeddings | $dataset"
    echo "-----------------------------------------------------------------------------------------------------------"
    
    # 3. Run standalone literal model
    literal_model="${base_model}_literal"
    echo ">>> Running STANDALONE LITERAL MODEL: $literal_model"
    
    exp_dir_standalone="${base_exp_dir}/LiteralE"
    mkdir -p "$exp_dir_standalone"
    
    cmd_standalone="python main.py \
      --dataset \"$dataset\" \
      --model \"$literal_model\" \
      --batch_size $batch_size \
      --num_iterations $num_iterations \
      --lr $lr \
      --eval_at_epochs 200 250 256 \
      --embedding_dim $embedding_dim \
      --exp_dir \"$exp_dir_standalone\"" 
    
    echo "Command: $cmd_standalone"
    eval $cmd_standalone
    echo "✓ Completed STANDALONE LITERAL: $literal_model | $dataset"
    echo "==========================================================================================================="
    echo ""
    
  done
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="