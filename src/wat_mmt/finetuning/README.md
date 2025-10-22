# IndicTrans2 Finetuning

This module provides a complete pipeline for finetuning the IndicTrans2 model on custom translation datasets with support for both **LoRA** and **full finetuning** approaches.

## Features

- ✅ **LoRA Finetuning**: Parameter-efficient training with minimal memory requirements
- ✅ **Full Finetuning**: Traditional full model training
- ✅ **Multilingual Training**: Train on all languages simultaneously
- ✅ **Flexible Data Source**: Choose between original or corrected translations
- ✅ **Automatic Evaluation**: BLEU score computation on dev set
- ✅ **Easy Inference**: Simple API for translation
- ✅ **Mac M2 Compatible**: Optimized for Apple Silicon

## Installation

Dependencies are already included in the project. Make sure you have:

```bash
uv add peft>=0.7.0 datasets>=2.14.0 sacrebleu>=2.3.0 evaluate>=0.4.0
```

## Quick Start

### 1. Training

#### LoRA Finetuning (Recommended)

```bash
# Train with corrected data (default)
python -m src.wat_mmt.finetuning.finetune \
    --train-data combined_processed_data/combined_results.csv \
    --dev-data combined_data/combined_dev.csv \
    --method lora \
    --use-corrected \
    --output-dir models/indictrans2-lora-corrected \
    --num-epochs 3 \
    --batch-size 8

# Train with original data (for comparison)
python -m src.wat_mmt.finetuning.finetune \
    --train-data combined_processed_data/combined_results.csv \
    --dev-data combined_data/combined_dev.csv \
    --method lora \
    --use-original \
    --output-dir models/indictrans2-lora-original \
    --num-epochs 3 \
    --batch-size 8
```

#### Full Finetuning

```bash
python -m src.wat_mmt.finetuning.finetune \
    --train-data combined_processed_data/combined_results.csv \
    --dev-data combined_data/combined_dev.csv \
    --method full \
    --use-corrected \
    --output-dir models/indictrans2-full-corrected \
    --num-epochs 3 \
    --batch-size 4  # Lower batch size for full finetuning
```

### 2. Inference

#### Single Text Translation

```bash
python -m src.wat_mmt.finetuning.inference \
    --model-path models/indictrans2-lora-corrected \
    --text "white block on tower" \
    --target-lang hindi
```

#### Interactive Mode

```bash
python -m src.wat_mmt.finetuning.inference \
    --model-path models/indictrans2-lora-corrected \
    --interactive \
    --target-lang hindi
```

#### Batch Translation from File

```bash
python -m src.wat_mmt.finetuning.inference \
    --model-path models/indictrans2-lora-corrected \
    --input-file input.csv \
    --output-file output.csv \
    --target-lang bengali \
    --text-column english_caption \
    --batch-size 32
```

## Python API Usage

### Training

```python
from src.wat_mmt.finetuning import FinetuningConfig, IndicTrans2Finetuner

# Create configuration
config = FinetuningConfig(
    method="lora",
    use_corrected=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

# Create finetuner and train
finetuner = IndicTrans2Finetuner(config)
trainer, metrics = finetuner.train()

print(f"Final BLEU: {metrics['eval_bleu']:.2f}")
```

### Inference

```python
from src.wat_mmt.finetuning import IndicTrans2Translator

# Load model
translator = IndicTrans2Translator(
    model_path="models/indictrans2-lora-corrected",
    is_lora=True
)

# Translate single text
translation = translator.translate(
    text="white block on tower",
    target_lang="hindi"
)
print(translation)

# Translate multiple texts
translations = translator.translate(
    text=["Hello", "Good morning", "Thank you"],
    target_lang="bengali"
)

# Batch translation
large_texts = [...]  # List of 1000+ texts
translations = translator.translate_batch(
    texts=large_texts,
    target_lang="malayalam",
    batch_size=32
)
```

## Configuration Options

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train-data` | Path | `combined_processed_data/combined_results.csv` | Training data path |
| `--dev-data` | Path | `combined_data/combined_dev.csv` | Dev data path |
| `--use-corrected` | Flag | True | Use corrected translations |
| `--use-original` | Flag | False | Use original translations |
| `--method` | str | `lora` | Finetuning method: `lora` or `full` |
| `--output-dir` | Path | `models/indictrans2-finetuned` | Output directory |
| `--num-epochs` | int | 3 | Number of training epochs |
| `--batch-size` | int | 8 | Training batch size |
| `--learning-rate` | float | 3e-5 | Learning rate |

### LoRA-specific Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lora-r` | int | 16 | LoRA rank (higher = more capacity) |
| `--lora-alpha` | int | 32 | LoRA alpha (scaling factor) |
| `--lora-dropout` | float | 0.1 | LoRA dropout |

### Inference Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-path` | Path | Required | Path to finetuned model |
| `--target-lang` | str | `hindi` | Target language |
| `--max-length` | int | 256 | Maximum generation length |
| `--num-beams` | int | 5 | Number of beams for beam search |
| `--batch-size` | int | 16 | Batch size for file mode |

## Supported Languages

| Language | Language Code | FLORES-200 Code |
|----------|---------------|-----------------|
| Hindi | `hindi` | `hin_Deva` |
| Bengali | `bengali` | `ben_Beng` |
| Malayalam | `malayalam` | `mal_Mlym` |
| Odia | `odia` | `ory_Orya` |
| English | `english` | `eng_Latn` |

## Performance Tips

### For Mac M2

1. **Batch Size**: Start with 8 for LoRA, 4 for full finetuning
2. **Memory**: LoRA requires ~8-10GB RAM, full needs ~16GB+
3. **Speed**: LoRA training takes ~2-4 hours, full takes ~8-12 hours
4. **Inference**: Use `num_beams=5` for quality, `num_beams=1` for speed

### General Tips

1. **Start with LoRA**: It's faster and less prone to overfitting
2. **Use Corrected Data**: Generally provides better quality
3. **Monitor BLEU**: Should improve over training epochs
4. **Checkpoint Selection**: Use best checkpoint based on dev BLEU

## Directory Structure

```
src/wat_mmt/finetuning/
├── __init__.py           # Package initialization
├── config.py             # Configuration classes
├── data_processor.py     # Data loading and preprocessing
├── finetune.py          # Main training script
├── inference.py         # Translation inference
└── README.md            # This file

models/
├── indictrans2-lora-corrected/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   └── ...
└── indictrans2-full-corrected/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── ...
```

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` (try 4 or 2)
- Use LoRA instead of full finetuning
- Enable gradient checkpointing in `config.py`

### Low BLEU Scores

- Train for more epochs
- Increase LoRA rank (`--lora-r 32`)
- Check data quality
- Ensure you're using corrected translations

### Slow Training

- Increase batch size if memory allows
- Reduce evaluation frequency (`eval_steps`)
- Use fewer beams during evaluation

## Examples

See the [examples](../../../examples/) directory for complete notebooks and scripts.

## Citation

If you use this code, please cite the IndicTrans2 paper:

```bibtex
@inproceedings{gala2023indictrans,
  title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
  author={Gala, Jay and Chitale, Pranjal A and others},
  booktitle={TMLR},
  year={2023}
}
```

## License

This project follows the same license as the main repository.

