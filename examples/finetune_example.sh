#!/bin/bash

# Example script for finetuning IndicTrans2
# Run from the project root directory

set -e

echo "===================================="
echo "IndicTrans2 Finetuning Example"
echo "===================================="
echo ""

# 1. Train with LoRA and corrected data
echo "1. Training with LoRA and corrected data..."
python -m src.wat_mmt.finetuning.finetune \
    --train-data combined_processed_data/combined_results.csv \
    --dev-data combined_data/combined_dev.csv \
    --method lora \
    --use-corrected \
    --output-dir models/indictrans2-lora-corrected \
    --num-epochs 3 \
    --batch-size 8 \
    --learning-rate 3e-5 \
    --lora-r 16 \
    --lora-alpha 32

echo ""
echo "Training complete!"
echo ""

# 2. Test translation
echo "2. Testing translation..."
python -m src.wat_mmt.finetuning.inference \
    --model-path models/indictrans2-lora-corrected \
    --text "white block on tower" \
    --target-lang hindi

echo ""

python -m src.wat_mmt.finetuning.inference \
    --model-path models/indictrans2-lora-corrected \
    --text "the head of a girl" \
    --target-lang bengali

echo ""
echo "Example complete!"

