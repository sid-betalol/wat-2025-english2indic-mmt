"""Utility functions for data cleaning pipeline."""

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def load_image(image_path: str | Path) -> Image.Image:
    """Load an image from disk.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object
    """
    return Image.open(image_path).convert("RGB")


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string.

    Args:
        image: PIL Image object

    Returns:
        Base64 encoded string
    """
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_csv_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the combined dataset CSV.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with the dataset
    """
    return pd.read_csv(csv_path, on_bad_lines="warn", engine="python")


def save_checkpoint(
    checkpoint_path: str | Path, processed_indices: list[int], stats: dict
) -> None:
    """Save checkpoint of processed data.

    Args:
        checkpoint_path: Path to save checkpoint
        processed_indices: List of processed row indices
        stats: Statistics dictionary
    """
    checkpoint_data = {
        "processed_indices": processed_indices,
        "stats": stats,
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def load_checkpoint(checkpoint_path: str | Path) -> tuple[list[int], dict]:
    """Load checkpoint of processed data.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (processed_indices, stats)
    """
    if not Path(checkpoint_path).exists():
        return [], {}

    with open(checkpoint_path) as f:
        checkpoint_data = json.load(f)

    return checkpoint_data["processed_indices"], checkpoint_data["stats"]


def save_results(results: list[dict], output_path: str | Path) -> None:
    """Save cleaned results to CSV.

    Args:
        results: List of result dictionaries
        output_path: Path to save CSV
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


def save_statistics(stats: dict, output_path: str | Path) -> None:
    """Save statistics report to JSON.

    Args:
        stats: Statistics dictionary
        output_path: Path to save JSON
    """
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)


def get_language_ratio_info(language: str) -> str:
    """Get information about typical word count ratio for a language.

    Args:
        language: Language name (hindi, bengali, malayalam, odia)

    Returns:
        String describing typical word count ratio
    """
    ratios = {
        "hindi": "about the same number of words",
        "bengali": "about 20% fewer words",
        "malayalam": "about 25% fewer words",
        "odia": "about the same number of words",
    }
    return ratios.get(language.lower(), "a natural length")


def initialize_stats() -> dict[str, Any]:
    """Initialize statistics tracking dictionary.

    Returns:
        Empty statistics dictionary structure
    """
    return {
        "total_examples": 0,
        "per_language": {
            "hindi": {
                "total": 0,
                "missing_captions": 0,
                "correct": 0,
                "corrected": 0,
                "correction_reasons": {
                    "visual_context_needed": 0,
                    "poor_translation": 0,
                    "missing_caption": 0,
                },
            },
            "bengali": {
                "total": 0,
                "missing_captions": 0,
                "correct": 0,
                "corrected": 0,
                "correction_reasons": {
                    "visual_context_needed": 0,
                    "poor_translation": 0,
                    "missing_caption": 0,
                },
            },
            "malayalam": {
                "total": 0,
                "missing_captions": 0,
                "correct": 0,
                "corrected": 0,
                "correction_reasons": {
                    "visual_context_needed": 0,
                    "poor_translation": 0,
                    "missing_caption": 0,
                },
            },
            "odia": {
                "total": 0,
                "missing_captions": 0,
                "correct": 0,
                "corrected": 0,
                "correction_reasons": {
                    "visual_context_needed": 0,
                    "poor_translation": 0,
                    "missing_caption": 0,
                },
            },
        },
        "overall": {
            "total_corrections": 0,
            "visual_context_needed": 0,
            "poor_translation": 0,
            "missing_caption": 0,
        },
    }


def update_stats(
    stats: dict, language: str, was_corrected: bool, correction_reason: str
) -> None:
    """Update statistics with processing result.

    Args:
        stats: Statistics dictionary to update
        language: Language being processed
        was_corrected: Whether caption was corrected
        correction_reason: Reason for correction
    """
    lang_stats = stats["per_language"][language.lower()]
    lang_stats["total"] += 1

    if was_corrected:
        lang_stats["corrected"] += 1
        lang_stats["correction_reasons"][correction_reason] += 1
        stats["overall"]["total_corrections"] += 1
        stats["overall"][correction_reason] += 1
    else:
        lang_stats["correct"] += 1


def load_prompt_template(prompt_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        prompt_name: Name of the prompt file (e.g., 'judge_prompt.txt')

    Returns:
        Prompt template string
    """
    prompts_dir = Path(__file__).parent / "prompts"
    prompt_path = prompts_dir / prompt_name
    with open(prompt_path) as f:
        return f.read()
