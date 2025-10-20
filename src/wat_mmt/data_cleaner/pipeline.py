"""Main pipeline for data cleaning."""

import os
import tempfile
import time
from pathlib import Path
from typing import ClassVar

import dspy
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from .corrector import VisualCaptionCorrector
from .judge import CaptionJudge
from .translator import IndicTranslator
from .utils import (
    initialize_stats,
    load_checkpoint,
    load_csv_data,
    load_image,
    save_checkpoint,
    save_results,
    save_statistics,
    update_stats,
)


class DataCleaningPipeline:
    """Pipeline for cleaning multimodal translation dataset."""

    # Language columns in the CSV
    LANGUAGES: ClassVar[list[str]] = ["hindi", "bengali", "malayalam", "odia"]
    LANG_COLUMNS: ClassVar[dict[str, str]] = {
        "hindi": "hindi_text",
        "bengali": "bengali_text",
        "malayalam": "malayalam_text",
        "odia": "odia_text",
    }

    def __init__(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        output_dir: str | Path,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        judge_model: str | None = None,
        judge_provider: str | None = None,
        corrector_model: str | None = None,
        corrector_provider: str | None = None,
        checkpoint_frequency: int = 100,
        sample_size: int | None = None,
    ):
        """Initialize the data cleaning pipeline.

        Args:
            csv_path: Path to the combined CSV file
            images_dir: Directory containing cropped images
            output_dir: Directory to save outputs
            model: Default model name
                (used if judge/corrector not specified)
            provider: Default LLM provider
                (used if judge/corrector not specified)
            judge_model: Model for judge (if None, uses default model)
            judge_provider: Provider for judge
                (if None, uses default provider)
            corrector_model: Model for corrector
                (if None, uses default model)
            corrector_provider: Provider for corrector
                (if None, uses default provider)
            checkpoint_frequency: Save checkpoint every N examples
            sample_size: If set, only process first N examples (for testing)
        """
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_frequency = checkpoint_frequency
        self.sample_size = sample_size

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load environment variables
        load_dotenv()

        # Helper function to create LM instances
        def create_lm(model_name: str, provider_name: str) -> dspy.LM:
            provider_name = provider_name.lower()
            if provider_name == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY not found in environment. "
                        "Please create a .env file with your API key."
                    )
                return dspy.LM(
                    f"openai/{model_name}", api_key=api_key, max_tokens=1000
                )
            elif provider_name == "google":
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GEMINI_API_KEY not found in environment. "
                        "Please create a .env file with your API key."
                    )
                return dspy.LM(
                    f"gemini/{model_name}", api_key=api_key, max_tokens=1000
                )
            else:
                raise ValueError(
                    f"Unsupported provider: {provider_name}. "
                    "Choose 'openai' or 'google'."
                )

        # Determine models for each component
        judge_model_name = judge_model or model
        judge_provider_name = judge_provider or provider
        corrector_model_name = corrector_model or model
        corrector_provider_name = corrector_provider or provider

        # Create LM instances
        judge_lm = create_lm(judge_model_name, judge_provider_name)
        corrector_lm = create_lm(corrector_model_name, corrector_provider_name)

        # Configure default LM (for other modules like translator)
        default_lm = create_lm(model, provider)
        dspy.configure(lm=default_lm)

        # Initialize modules with specific LMs
        print("Initializing modules (cache enabled for efficiency)...")
        print(f"  Judge: {judge_provider_name}/{judge_model_name}")
        print(f"  Corrector: {corrector_provider_name}/{corrector_model_name}")

        self.judge = CaptionJudge(lm=judge_lm)
        self.corrector = VisualCaptionCorrector(lm=corrector_lm)
        self.translator = IndicTranslator()

        # Load data
        print(f"Loading data from {self.csv_path}...")
        self.df = load_csv_data(self.csv_path)
        if self.sample_size:
            self.df = self.df.head(self.sample_size)
            print(f"Using sample of {self.sample_size} examples")

        # Initialize tracking
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.processed_indices, self.stats = load_checkpoint(
            self.checkpoint_path
        )
        if not self.stats:
            self.stats = initialize_stats()
            self.stats["total_examples"] = len(self.df)

        self.results = []

        print("Pipeline initialized successfully!")

    def process_caption(
        self,
        image: Image.Image | None,
        image_path: Path,
        english_caption: str,
        target_caption: str,
        language: str,
        temp_image_path: Path | None,
    ) -> dict:
        """Process a single caption for a language.

        Args:
            image: Pre-loaded PIL Image (or None if not found)
            image_path: Path to image file (for error messages)
            english_caption: English caption
            target_caption: Target language caption to evaluate
            language: Target language name
            temp_image_path: Pre-created temp image file path
                (None if image not found; optimization to reuse across ops)

        Returns:
            Dictionary with processing results
        """
        # Check if image was loaded successfully
        if image is None:
            print(f"Warning: Image not found: {image_path}")
            return {
                "original_target_caption": target_caption,
                "corrected_target_caption": target_caption,
                "was_corrected": False,
                "correction_reason": "none",
                "judge_confidence": 0.0,
                "judge_explanation": "Image file not found",
            }

        # Step 1: Check if caption is missing
        is_missing = (
            pd.isna(target_caption)
            or str(target_caption).strip() == ""
            or str(target_caption).strip().lower() in ["nan", "none", "null"]
        )

        if is_missing:
            # Route directly to LLM corrector
            correction = self.corrector(
                image=image,
                english_caption=english_caption,
                target_language=language,
                original_target_caption="",
                temp_image_path=temp_image_path,
            )
            return {
                "original_target_caption": "",
                "corrected_target_caption": correction.corrected_caption,
                "was_corrected": True,
                "correction_reason": "missing_caption",
                "judge_confidence": 1.0,
                "judge_explanation": "Caption was missing",
                "correction_explanation": correction.explanation,
            }

        # Step 2: Judge the caption
        judgment = self.judge(
            image=image,
            english_caption=english_caption,
            target_caption=str(target_caption),
            target_language=language,
            temp_image_path=temp_image_path,
        )

        # Step 3: Route based on judgment
        if judgment.status == "correct":
            # Keep original
            return {
                "original_target_caption": target_caption,
                "corrected_target_caption": target_caption,
                "was_corrected": False,
                "correction_reason": "none",
                "judge_confidence": float(judgment.confidence),
                "judge_explanation": judgment.explanation,
            }

        # Skip correction if confidence is too low (uncertain judgments)
        if judgment.status == "incorrect" and float(judgment.confidence) < 0.7:
            return {
                "original_target_caption": target_caption,
                "corrected_target_caption": target_caption,
                "was_corrected": False,
                "correction_reason": "low_confidence_skip",
                "judge_confidence": float(judgment.confidence),
                "judge_explanation": (
                    f"Flagged as {judgment.reason} but low confidence - "
                    f"keeping original. {judgment.explanation}"
                ),
            }

        elif judgment.reason == "visual_context_needed":
            # Use LLM corrector (reuse temp file path)
            correction = self.corrector(
                image=image,
                english_caption=english_caption,
                target_language=language,
                original_target_caption=str(target_caption),
                temp_image_path=temp_image_path,
            )
            return {
                "original_target_caption": target_caption,
                "corrected_target_caption": correction.corrected_caption,
                "was_corrected": True,
                "correction_reason": "visual_context_needed",
                "judge_confidence": float(judgment.confidence),
                "judge_explanation": judgment.explanation,
                "correction_explanation": correction.explanation,
            }

        elif judgment.reason == "poor_translation":
            # Use IndicTrans2
            translation = self.translator.translate(
                english_caption, target_language=language
            )
            return {
                "original_target_caption": target_caption,
                "corrected_target_caption": translation,
                "was_corrected": True,
                "correction_reason": "poor_translation",
                "judge_confidence": float(judgment.confidence),
                "judge_explanation": judgment.explanation,
                "correction_explanation": "Retranslated using IndicTrans2",
            }

        else:
            # Fallback: keep original
            return {
                "original_target_caption": target_caption,
                "corrected_target_caption": target_caption,
                "was_corrected": False,
                "correction_reason": "none",
                "judge_confidence": float(judgment.confidence),
                "judge_explanation": judgment.explanation,
            }

    def process_row(self, idx: int, row: pd.Series) -> dict:
        """Process a single row (all languages).

        Args:
            idx: Row index
            row: DataFrame row

        Returns:
            Single dictionary with all languages as columns
        """
        # Start with the common fields
        combined_result = {
            "image_id": row["image_id"],
            "x": row["x"],
            "y": row["y"],
            "width": row["width"],
            "height": row["height"],
            "english_caption": row["english_text"],
        }

        # OPTIMIZATION: Load image once for all languages
        image_id = str(row["image_id"])
        image_path = self.images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            # Try other common extensions
            for ext in [".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                alt_path = self.images_dir / f"{image_id}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break

        # Load image once (or None if not found)
        image = load_image(image_path) if image_path.exists() else None

        # OPTIMIZATION: Create temp file once for all 4 languages
        # (only if image exists)
        temp_image_path = None
        if image is not None:
            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False
            ) as tmp_file:
                temp_image_path = Path(tmp_file.name)
                image.save(temp_image_path, format="PNG")

        try:
            # Process each language and add as prefixed columns
            for language in self.LANGUAGES:
                lang_col = self.LANG_COLUMNS[language]
                target_caption = row[lang_col]

                result = self.process_caption(
                    image=image,
                    image_path=image_path,
                    english_caption=row["english_text"],
                    target_caption=target_caption,
                    language=language,
                    temp_image_path=temp_image_path,
                )

                # Add language-specific columns with prefix
                prefix = language
                combined_result[f"{prefix}_original"] = result[
                    "original_target_caption"
                ]
                combined_result[f"{prefix}_corrected"] = result[
                    "corrected_target_caption"
                ]
                combined_result[f"{prefix}_was_corrected"] = result[
                    "was_corrected"
                ]
                combined_result[f"{prefix}_reason"] = result[
                    "correction_reason"
                ]
                combined_result[f"{prefix}_confidence"] = result[
                    "judge_confidence"
                ]
                combined_result[f"{prefix}_explanation"] = result[
                    "judge_explanation"
                ]

                # Update stats
                update_stats(
                    self.stats,
                    language,
                    result["was_corrected"],
                    result["correction_reason"],
                )

            return combined_result
        finally:
            # Clean up temp file (created once for all languages)
            if temp_image_path and temp_image_path.exists():
                temp_image_path.unlink(missing_ok=True)

    def run(self) -> None:
        """Run the complete pipeline."""
        print(f"\nProcessing {len(self.df)} examples...")
        print(f"Already processed: {len(self.processed_indices)} indices")

        # Progress bar
        pbar = tqdm(
            total=len(self.df),
            initial=len(self.processed_indices),
            desc="Processing",
        )

        for idx, row in self.df.iterrows():
            # Skip if already processed
            if idx in self.processed_indices:
                continue

            try:
                # Process row with timing
                start_time = time.time()
                row_result = self.process_row(idx, row)
                elapsed = time.time() - start_time

                self.results.append(row_result)

                # Mark as processed
                self.processed_indices.append(idx)

                # Show timing for first few examples
                if len(self.processed_indices) <= 3:
                    print(
                        f"\n  Row {idx}: {elapsed:.2f}s "
                        f"(~{elapsed / 4:.2f}s per language)"
                    )

                # Save checkpoint
                if len(self.processed_indices) % self.checkpoint_frequency == 0:
                    save_checkpoint(
                        self.checkpoint_path, self.processed_indices, self.stats
                    )
                    # Also save intermediate results
                    save_results(
                        self.results, self.output_dir / "results_partial.csv"
                    )

                pbar.update(1)

            except Exception as e:
                print(f"\nError processing row {idx}: {e}")
                # Save checkpoint on error
                save_checkpoint(
                    self.checkpoint_path, self.processed_indices, self.stats
                )
                raise

        pbar.close()

        # Save final results
        print("\nSaving final results...")
        save_results(self.results, self.output_dir / "cleaned_data.csv")
        save_statistics(self.stats, self.output_dir / "statistics.json")

        # Print summary
        self._print_summary()

        print(f"\n✅ Pipeline complete! Results saved to {self.output_dir}")

    def _print_summary(self) -> None:
        """Print statistics summary."""
        print("\n" + "=" * 60)
        print("CLEANING STATISTICS SUMMARY")
        print("=" * 60)

        print(f"\nTotal examples processed: {self.stats['total_examples']}")
        print(
            f"Total corrections: {self.stats['overall']['total_corrections']}"
        )
        print(
            f"  - Visual context needed: "
            f"{self.stats['overall']['visual_context_needed']}"
        )
        print(
            f"  - Poor translation: {self.stats['overall']['poor_translation']}"
        )
        print(
            f"  - Missing caption: {self.stats['overall']['missing_caption']}"
        )

        print("\nPer-Language Breakdown:")
        print("-" * 60)

        for lang in self.LANGUAGES:
            lang_stats = self.stats["per_language"][lang]
            total = lang_stats["total"]
            corrected = lang_stats["corrected"]
            correct = lang_stats["correct"]
            correction_pct = (corrected / total * 100) if total > 0 else 0

            print(f"\n{lang.upper()}:")
            print(f"  Total: {total}")
            print(f"  Correct: {correct} ({100 - correction_pct:.1f}%)")
            print(f"  Corrected: {corrected} ({correction_pct:.1f}%)")
            reasons = lang_stats["correction_reasons"]
            print(f"    - Visual context: {reasons['visual_context_needed']}")
            print(f"    - Poor translation: {reasons['poor_translation']}")
            print(f"    - Missing caption: {reasons['missing_caption']}")

        print("\n" + "=" * 60)
