# WAT 2025 English→Indic Multimodal Translation

Complete pipeline for the WAT 2025 English-to-Indic Multimodal Machine Translation task, featuring:
- **Automated Data Cleaning**: LLM-as-a-judge methodology for caption quality assessment
- **IndicTrans2 Finetuning**: training pipeline with LoRA and full finetuning support

## Quick Start

### Data Cleaning Pipeline

```bash
# Clone and setup
git clone https://github.com/sid-betalol/wat-2025-english2indic-mmt.git
cd wat-2025-english2indic-mmt
uv sync

# Set up API keys
cp env.example .env
# Edit .env and add your GEMINI_API_KEY, OPENAI_API_KEY, and HuggingFace token

# Login to HuggingFace (for IndicTrans2 model)
uv run huggingface-cli login

# Test with 10 examples
uv run clean_data.py --sample 10

# Process full training set
uv run clean_data.py
```

### Model Finetuning

```bash
# Train IndicTrans2 with LoRA (recommended)
uv run python -m src.wat_mmt.finetuning.finetune \
    --method lora \
    --use-corrected \
    --output-dir models/my-model

# Translate text
uv run python -m src.wat_mmt.finetuning.inference \
    --model-path models/my-model \
    --text "Hello world" \
    --target-lang hindi
```

See [FINETUNING_QUICKSTART.md](FINETUNING_QUICKSTART.md) for detailed finetuning guide.

## How It Works

### Data Cleaning Pipeline

The pipeline cleans multimodal translation data across 4 languages (Hindi, Bengali, Malayalam, Odia) using a 3-stage approach:

#### 1. Judge Stage
- Gemini 2.5 Flash Lite evaluates each caption with the image
- Classifies as: `correct`, `visual_context_needed`, or `poor_translation`

#### 2. Correction Routing
- **Missing/Visual Context** → GPT-4o-mini regenerates caption using image
- **Poor Translation** → IndicTrans2 retranslates from English
- **Correct** → Keep original

**Note**: You can use any suitable OpenAI or Gemini models for judge/corrector based on your API access and preferences.

#### 3. Output
One row per image with all 4 languages:
- Original and corrected captions for each language
- Correction flags, reasons, confidence scores
- Judge explanations

### Model Finetuning

The finetuning pipeline trains IndicTrans2 on cleaned translation data:

#### LoRA Finetuning (Recommended)
- **Parameter-efficient**: Only trains ~0.1% of model parameters
- **Fast**: 2-4 hours on M2 Mac, faster on GPUs
- **Memory-efficient**: ~8-10GB RAM/VRAM
- **Small checkpoints**: ~10-50MB adapter files

#### Full Finetuning
- **Traditional approach**: Trains all model parameters
- **Higher quality**: Slightly better performance
- **Resource-intensive**: ~16GB+ RAM/VRAM, 8-12 hours
- **Large checkpoints**: ~800MB model files

#### Features
- Multilingual training on all 4 languages simultaneously
- Automatic BLEU score evaluation on dev set
- Choice between original or corrected translations
- Multi-GPU support for faster training
- Simple Python API for inference

## Complete Workflow

### 1. Data Preparation

```bash
# Combine datasets from all languages
uv run combine_datasets.py

# Crop images to bounding boxes
uv run crop_images.py
```

### 2. Data Cleaning

```bash
# Clean captions with LLM-as-a-judge
uv run clean_data.py

# Or test with a sample first
uv run clean_data.py --sample 100
```

### 3. Model Training

```bash
# Train IndicTrans2 with LoRA on cleaned data
uv run python -m src.wat_mmt.finetuning.finetune \
    --method lora \
    --use-corrected \
    --output-dir models/indictrans2-lora

# For comparison, train on original data
uv run python -m src.wat_mmt.finetuning.finetune \
    --method lora \
    --use-original \
    --output-dir models/indictrans2-lora-original
```

### 4. Inference

```bash
# Translate with your finetuned model
uv run python -m src.wat_mmt.finetuning.inference \
    --model-path models/indictrans2-lora \
    --input-file combined_data/combined_test.csv \
    --output-file results/predictions.csv \
    --target-lang hindi \
    --batch-size 32
```

## Command Options

### Data Cleaning Options

```bash
uv run clean_data.py [OPTIONS]

--sample N              # Process only first N examples (testing)
--judge-model MODEL     # Judge model (default: gemini-2.5-flash-lite)
--corrector-model MODEL # Corrector model (default: gpt-4o-mini)
--csv PATH              # Input CSV (default: combined_data/combined_train.csv)
--images PATH           # Images directory (default: cropped_images/train)
--output PATH           # Output directory (default: cleaned_data)
--checkpoint-freq N     # Save checkpoint every N examples (default: 100)
--max-concurrent N      # Max concurrent API calls (default: 4)
```

**Note**: You can use any OpenAI or Gemini models for judge/corrector. Examples: `gpt-4o`, `gpt-4o-mini`, `gemini-2.0-flash-exp`, `gemini-2.5-flash-lite`

### Finetuning Options

```bash
uv run python -m src.wat_mmt.finetuning.finetune [OPTIONS]

# Training method and data
--method {lora,full}    # Training method (default: lora)
--use-corrected         # Use corrected translations (default)
--use-original          # Use original translations

# Model and data paths
--train-data PATH       # Training data CSV
--dev-data PATH         # Dev data CSV
--output-dir PATH       # Output directory for model

# Training hyperparameters
--num-epochs N          # Number of epochs (default: 3)
--batch-size N          # Batch size (default: 8)
--learning-rate FLOAT   # Learning rate (default: 3e-5)

# LoRA-specific
--lora-r N              # LoRA rank (default: 16)
--lora-alpha N          # LoRA alpha (default: 32)

# Multi-GPU
--multi-gpu             # Enable multi-GPU training
```

### Inference Options

```bash
uv run python -m src.wat_mmt.finetuning.inference [OPTIONS]

--model-path PATH       # Path to finetuned model (required)
--target-lang LANG      # Target language: hindi, bengali, malayalam, odia

# Single text mode
--text "TEXT"           # Translate single text

# Interactive mode
--interactive           # Interactive translation mode

# Batch mode
--input-file PATH       # Input CSV/JSON file
--output-file PATH      # Output CSV/JSON file
--text-column NAME      # Column name for input text
--batch-size N          # Batch size (default: 16)

# Generation parameters
--max-length N          # Max generation length (default: 256)
--num-beams N           # Beam search size (default: 5)
```

## Output Format

**cleaned_data.csv**: Wide format with columns:
```
image_id, x, y, width, height, english_caption,
hindi_original, hindi_corrected, hindi_was_corrected, hindi_reason, hindi_confidence, hindi_explanation,
bengali_original, bengali_corrected, bengali_was_corrected, bengali_reason, bengali_confidence, bengali_explanation,
malayalam_original, malayalam_corrected, malayalam_was_corrected, malayalam_reason, malayalam_confidence, malayalam_explanation,
odia_original, odia_corrected, odia_was_corrected, odia_reason, odia_confidence, odia_explanation
```

**statistics.json**: Correction counts and percentages by language

## Features

### Data Cleaning
✅ **Automatic checkpointing** - Resume from interruptions  
✅ **DSPy caching** - Fast re-runs on same data  
✅ **MPS acceleration** - Optimized for Apple Silicon  
✅ **Progress tracking** - Real-time progress bars and timing  
✅ **Error handling** - Robust error recovery  
✅ **Concurrent processing** - Parallel API calls with rate limiting

### Model Finetuning
✅ **LoRA & Full Finetuning** - Choose your training approach  
✅ **Flexible Data Selection** - Train on original or corrected translations  
✅ **Multilingual Training** - All 4 languages in one model  
✅ **Automatic Evaluation** - BLEU scores on dev set  
✅ **Multi-GPU Support** - Faster training with multiple GPUs  
✅ **Easy Inference** - Simple Python API and CLI  
✅ **Mac M2 Optimized** - Works great on Apple Silicon

## Performance & Concurrency

The pipeline uses **two-level concurrent processing** for maximum throughput:

### Level 1: Row-Level Parallelism
- Multiple image rows processed simultaneously
- Uses `asyncio.TaskGroup` for concurrent execution
- Progress tracked with `tqdm.asyncio` for real-time updates

### Level 2: Language-Level Parallelism
- Within each row, all 4 languages processed concurrently
- Each language independently calls judge/corrector APIs

### Rate Limiting
- **Semaphore-based control** wraps each API call to prevent overload
- `--max-concurrent N` limits concurrent API calls across all rows/languages (default: 4)
- Example: Processing 2 rows with 4 languages each → 8 concurrent language tasks, but only 4 API calls active at once
- Each language task may make 1-2 API calls (judge + optional corrector), queued via semaphore

### Performance Tips
- **Increase concurrency** for faster processing: `--max-concurrent 8`
- **Respect rate limits**: Check your API provider's limits
- **Monitor costs**: More concurrency = faster but same total API calls
- **Typical speed**: ~10-15 seconds per row with 4 concurrent tasks

## Project Structure

```
├── data/                       # Raw datasets (4 languages)
├── combined_data/              # Combined CSVs
├── cropped_images/             # Cropped image regions
├── cleaned_data/               # Cleaned output
├── models/                     # Finetuned models
├── src/wat_mmt/
│   ├── data_cleaner/          # Cleaning pipeline
│   │   ├── judge.py           # LLM judge module
│   │   ├── corrector.py       # LLM corrector module
│   │   ├── translator.py      # IndicTrans2 wrapper
│   │   ├── pipeline.py        # Main orchestration
│   │   └── utils.py           # Helper functions
│   └── finetuning/            # Finetuning pipeline
│       ├── config.py          # Configuration classes
│       ├── data_processor.py  # Data loading & preprocessing
│       ├── finetune.py        # Training script
│       ├── inference.py       # Translation inference
│       └── README.md          # Detailed finetuning docs
├── examples/                   # Example scripts
│   ├── finetune_example.sh    # Training example
│   └── inference_example.py   # Inference example
├── clean_data.py              # Data cleaning entry point
├── combine_datasets.py        # Data preparation
├── crop_images.py             # Image preprocessing
└── FINETUNING_QUICKSTART.md   # Finetuning quick start guide
```

## Requirements

- Python 3.11+
- **For data cleaning**: 
  - Gemini API key (for judge - Gemini 2.5 Flash Lite by default)
  - OpenAI API key (for corrector - GPT-4o-mini by default)
  - *Note: You can use any suitable OpenAI or Gemini models for either role*
- HuggingFace account with access to `ai4bharat/indictrans2-en-indic-dist-200M`
- 8GB+ RAM for data cleaning and LoRA finetuning
- 16GB+ RAM for full finetuning
- Optional: CUDA GPU(s) for faster training

## Tips

### Data Cleaning
- **Test first**: Always use `--sample` to validate before full run
- **Monitor costs**: 
  - Gemini 2.5 Flash Lite: Very cost-effective for judge stage
  - GPT-4o-mini: ~$0.15/$0.60 per 1M tokens (input/output) for corrections
  - You can experiment with different model combinations based on cost/quality tradeoffs
- **Resume anytime**: Ctrl+C to stop, same command to resume
- **Check stats**: Review `statistics.json` for correction patterns

### Model Finetuning
- **Start with LoRA**: Faster, more efficient, and less prone to overfitting
- **Use corrected data**: Generally provides better translation quality
- **Monitor BLEU**: Should improve over training epochs
- **Batch size**: Start with 8 for LoRA, 4 for full finetuning, adjust based on memory
- **Multi-GPU**: Use `--multi-gpu` flag for 2+ GPUs to speed up training

## Citation

```bibtex
@inproceedings{betala-chokshi-2024-brotherhood,
    title = "Brotherhood at {WMT} 2024: Leveraging {LLM}-Generated Contextual Conversations for Cross-Lingual Image Captioning",
    author = "Betala, Siddharth and Chokshi, Ishan",
    booktitle = "Proceedings of the Ninth Conference on Machine Translation",
    year = "2024",
    publisher = "Association for Computational Linguistics",
}
```
