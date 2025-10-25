#!/bin/bash

# Quick test script to verify training works
# Runs only a few steps to test GPU usage and checkpoint saving

set -e

echo "================================================================"
echo "IndicTrans2 Training Test (Quick)"
echo "================================================================"
echo "This will run a quick test with only 10 steps to verify everything works"
echo ""

# Create test directory
mkdir -p test_models

# Create custom temp directory to avoid disk space issues
mkdir -p /home/kushan/tmp

# Set environment variables to handle disk space
export TMPDIR="/home/kushan/tmp"
export TEMP="/home/kushan/tmp"
export TMP="/home/kushan/tmp"

# Create temp directory if it doesn't exist
mkdir -p /home/kushan/tmp

# Test 1: LoRA Multi-GPU (quick test)
echo "================================================================="
echo "TEST 1: LoRA Multi-GPU (10 steps)"
echo "================================================================="

echo "Starting LoRA test..."
torchrun --nproc_per_node=2 --master_port=29500 -m src.wat_mmt.finetuning.finetune \
    --method lora \
    --use-corrected \
    --multi-gpu \
    --batch-size 32 \
    --max-steps 10 \
    --output-dir test_models/lora-test

echo ""
echo "✅ LoRA test completed!"

# Wait a moment
sleep 5

# Test 2: Full Multi-GPU (quick test)
echo ""
echo "================================================================="
echo "TEST 2: Full Multi-GPU (10 steps)"
echo "================================================================="

echo "Starting Full test..."
torchrun --nproc_per_node=2 --master_port=29501 -m src.wat_mmt.finetuning.finetune \
    --method full \
    --use-corrected \
    --multi-gpu \
    --batch-size 24 \
    --max-steps 10 \
    --output-dir test_models/full-test

echo ""
echo "✅ Full test completed!"

echo ""
echo "================================================================="
echo "TEST COMPLETE"
echo "================================================================="
echo "Both tests passed! Your setup is ready for full training."
echo ""
echo "To run the full training, use:"
echo "  ./run_training.sh"
echo ""
echo "Expected full training times:"
echo "  - LoRA Multi-GPU: 1-1.5 hours"
echo "  - Full Multi-GPU: 3-4 hours"
