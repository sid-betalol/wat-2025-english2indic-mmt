# WAT 2025 English→Indic Multimodal Translation

Automated data cleaning pipeline for the WAT 2025 English-to-Indic Multimodal Machine Translation task using LLM-as-a-judge methodology.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/sid-betalol/wat-2025-english2indic-mmt.git
cd wat-2025-english2indic-mmt
uv sync

# Set up API keys
cp env.example .env
# Edit .env and add your OPENAI_API_KEY and HuggingFace token

# Login to HuggingFace (for IndicTrans2 model)
uv run huggingface-cli login

# Test with 10 examples
uv run clean_data.py --sample 10

# Process full training set
uv run clean_data.py
```

## How It Works

The pipeline cleans multimodal translation data across 4 languages (Hindi, Bengali, Malayalam, Odia) using a 3-stage approach:

### 1. Judge Stage
- GPT-4o-mini evaluates each caption with the image
- Classifies as: `correct`, `visual_context_needed`, or `poor_translation`

### 2. Correction Routing
- **Missing/Visual Context** → LLM regenerates caption using image
- **Poor Translation** → IndicTrans2 retranslates from English
- **Correct** → Keep original

### 3. Output
One row per image with all 4 languages:
- Original and corrected captions for each language
- Correction flags, reasons, confidence scores
- Judge explanations

## Data Preparation

```bash
# 1. Combine datasets
uv run combine_datasets.py

# 2. Crop images to bounding boxes
uv run crop_images.py

# 3. Clean captions
uv run clean_data.py
```

## Command Options

```bash
uv run clean_data.py [OPTIONS]

--sample N              # Process only first N examples (testing)
--model MODEL           # OpenAI model (default: gpt-4o-mini)
--csv PATH              # Input CSV (default: combined_data/combined_train.csv)
--images PATH           # Images directory (default: cropped_images/train)
--output PATH           # Output directory (default: cleaned_data)
--checkpoint-freq N     # Save checkpoint every N examples (default: 100)
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

✅ **Automatic checkpointing** - Resume from interruptions  
✅ **DSPy caching** - Fast re-runs on same data  
✅ **MPS acceleration** - Optimized for Apple Silicon  
✅ **Progress tracking** - Real-time progress bars and timing  
✅ **Error handling** - Robust error recovery

## Project Structure

```
├── data/                       # Raw datasets (4 languages)
├── combined_data/              # Combined CSVs
├── cropped_images/             # Cropped image regions
├── cleaned_data/               # Cleaned output
├── src/wat_mmt/data_cleaner/  # Cleaning pipeline
│   ├── judge.py               # LLM judge module
│   ├── corrector.py           # LLM corrector module
│   ├── translator.py          # IndicTrans2 wrapper
│   ├── pipeline.py            # Main orchestration
│   └── utils.py               # Helper functions
├── clean_data.py              # Main entry point
├── combine_datasets.py        # Data preparation
└── crop_images.py             # Image preprocessing
```

## Requirements

- Python 3.11+
- OpenAI API key (for GPT-4o-mini)
- HuggingFace account with access to `ai4bharat/indictrans2-en-indic-dist-200M`
- 8GB+ RAM recommended for IndicTrans2 model

## Tips

- **Test first**: Always use `--sample` to validate before full run
- **Monitor costs**: GPT-4o-mini costs ~$0.15/$0.60 per 1M tokens (input/output)
- **Resume anytime**: Ctrl+C to stop, same command to resume
- **Check stats**: Review `statistics.json` for correction patterns

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
