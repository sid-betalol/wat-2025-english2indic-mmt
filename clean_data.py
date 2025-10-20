#!/usr/bin/env python3
"""CLI script for cleaning the WAT MMT dataset.

This script processes the multimodal translation dataset to identify and correct
translation errors using LLM-as-a-judge and automatic correction methods.

Usage:
    uv run clean_data.py --sample 20  # Test with 20 examples
    uv run clean_data.py              # Process full training set
"""

import argparse
from pathlib import Path

from src.wat_mmt.data_cleaner.pipeline import DataCleaningPipeline


def main():
    """Main entry point for the data cleaning script."""
    parser = argparse.ArgumentParser(
        description="Clean WAT MMT dataset using LLM-as-a-judge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="combined_data/combined_train.csv",
        help="Path to the combined CSV file",
    )

    parser.add_argument(
        "--images",
        type=str,
        default="cropped_images/train",
        help="Directory containing cropped images",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="cleaned_data",
        help="Output directory for cleaned data and statistics",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (e.g., gpt-4o-mini, gemini-1.5-flash, gemini-1.5-pro)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "google"],
        help="LLM provider to use",
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100,
        help="Save checkpoint every N examples",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only first N examples (for testing)",
    )

    args = parser.parse_args()

    # Convert to absolute paths
    workspace_root = Path(__file__).parent
    csv_path = workspace_root / args.csv
    images_dir = workspace_root / args.images
    output_dir = workspace_root / args.output

    # Validate inputs
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return 1

    if not images_dir.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        return 1

    # Print configuration
    print("=" * 60)
    print("WAT MMT DATA CLEANING PIPELINE")
    print("=" * 60)
    print(f"CSV file: {csv_path}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Checkpoint frequency: {args.checkpoint_freq}")
    if args.sample:
        print(f"Sample size: {args.sample} (testing mode)")
    print("=" * 60)

    # Initialize and run pipeline
    try:
        pipeline = DataCleaningPipeline(
            csv_path=csv_path,
            images_dir=images_dir,
            output_dir=output_dir,
            model=args.model,
            provider=args.provider,
            checkpoint_frequency=args.checkpoint_freq,
            sample_size=args.sample,
        )

        pipeline.run()

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("Progress has been saved. Run again to resume.")
        return 130

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
