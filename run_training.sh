#!/bin/bash

# Complete Training Script for IndicTrans2 on 2 GPUs
# Handles disk space issues and runs both LoRA and full finetuning

set -e

echo "================================================================"
echo "IndicTrans2 Complete Training Pipeline"
echo "================================================================"
echo "Start time: $(date)"
echo ""

# Create models directory
mkdir -p models

# Create custom temp directory to avoid disk space issues
mkdir -p /home/kushan/tmp

# Set timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to run training with proper error handling
run_training() {
    local method=$1
    local multi_gpu=$2
    local batch_size=$3
    local output_dir=$4
    local epochs=$5
    
    echo ""
    echo "================================================================="
    echo "Starting $method training"
    echo "================================================================="
    echo "Multi-GPU: $multi_gpu"
    echo "Batch size: $batch_size"
    echo "Output directory: $output_dir"
    echo "Epochs: $epochs"
    echo ""
    
    # Prepare command
    if [ "$multi_gpu" = "true" ]; then
        local cmd="torchrun --nproc_per_node=2 --master_port=29500 -m src.wat_mmt.finetuning.finetune"
    else
        local cmd="uv run python -m src.wat_mmt.finetuning.finetune"
    fi
    
    cmd="$cmd --method $method"
    cmd="$cmd --use-corrected"
    cmd="$cmd --output-dir $output_dir"
    cmd="$cmd --num-epochs $epochs"
    cmd="$cmd --batch-size $batch_size"
    
    if [ "$multi_gpu" = "true" ]; then
        cmd="$cmd --multi-gpu"
    fi
    
    echo "Command: $cmd"
    echo ""
    
    # Set environment variables to handle disk space
    export TMPDIR="/home/kushan/tmp"
    export TEMP="/home/kushan/tmp"
    export TMP="/home/kushan/tmp"
    
    # Create temp directory if it doesn't exist
    mkdir -p /home/kushan/tmp
    
    # Run training
    local start_time=$(date +%s)
    
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        
        echo ""
        echo "✅ Training completed successfully!"
        echo "Duration: ${hours}h ${minutes}m"
        echo "Output saved to: $output_dir"
        return 0
    else
        echo ""
        echo "❌ Training failed!"
        return 1
    fi
}

# Training 1: Full Multi-GPU
echo "================================================================="
echo "TRAINING 2/2: Full Multi-GPU"
echo "================================================================="
run_training "full" "true" "16" "models/indictrans2-full-2gpu-$TIMESTAMP" "3"

# Wait between trainings
echo ""
echo "Waiting 30 seconds before next training..."
sleep 30

# Training 2: LoRA Multi-GPU
echo ""
echo "================================================================="
echo "TRAINING 1/2: LoRA Multi-GPU"
echo "================================================================="
run_training "lora" "true" "16" "models/indictrans2-lora-2gpu-$TIMESTAMP" "3"



# Final summary
echo ""
echo "================================================================="
echo "TRAINING PIPELINE COMPLETE"
echo "================================================================="
echo "End time: $(date)"
echo ""
echo "Models saved in:"
echo "  - models/indictrans2-lora-2gpu-$TIMESTAMP/"
echo "  - models/indictrans2-full-2gpu-$TIMESTAMP/"
echo ""
echo "🎉 Both training runs completed!"
echo "Check the models directory for your trained models."
