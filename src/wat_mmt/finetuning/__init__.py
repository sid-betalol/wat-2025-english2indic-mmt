"""IndicTrans2 Finetuning Module.

This module provides utilities for finetuning the IndicTrans2 model
on custom translation datasets with support for both LoRA and full
finetuning approaches.
"""

from .config import FinetuningConfig, LoRAConfig
from .data_processor import TranslationDataProcessor
from .inference import IndicTrans2Translator

__all__ = [
    "FinetuningConfig",
    "IndicTrans2Translator",
    "LoRAConfig",
    "TranslationDataProcessor",
]
