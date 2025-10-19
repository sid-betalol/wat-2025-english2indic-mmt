# Data Cleaner Module

Internal implementation of the data cleaning pipeline.

## Modules

- **`judge.py`**: DSPy module that evaluates caption quality using GPT-4o-mini with image context
- **`corrector.py`**: DSPy module that generates corrected captions using visual context
- **`translator.py`**: IndicTrans2 wrapper with Apple Silicon MPS support
- **`pipeline.py`**: Main orchestration logic with checkpointing
- **`utils.py`**: Helper functions for image loading, CSV handling, checkpointing
- **`prompts/`**: Prompt templates for LLM modules

## Architecture

```
DataCleaningPipeline
├── CaptionJudge (judge.py)
│   └── Evaluates: correct | visual_context_needed | poor_translation
├── VisualCaptionCorrector (corrector.py)
│   └── Generates corrected captions using image + English
└── IndicTranslator (translator.py)
    └── Retranslates using IndicTrans2 model
```

## Key Features

- **Lazy loading**: IndicTrans2 only loads when needed
- **Automatic checkpointing**: Saves progress every 100 examples
- **Device detection**: Auto-selects MPS/CUDA/CPU
- **Error recovery**: Saves checkpoint on errors
- **DSPy integration**: Structured prompting with type safety

## Usage

See main README for usage instructions. This module is used internally by `clean_data.py`.
