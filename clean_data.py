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
        help="Default model name (used if --judge-model or --corrector-model not specified)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "google"],
        help="Default LLM provider (used if --judge-provider or --corrector-provider not specified)",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model for judge (e.g., gpt-4o-mini, gemini-1.5-flash). If not set, uses --model",
    )

    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        choices=["openai", "google"],
        help="Provider for judge. If not set, uses --provider",
    )

    parser.add_argument(
        "--corrector-model",
        type=str,
        default=None,
        help="Model for corrector (e.g., gpt-4o, gemini-1.5-pro). If not set, uses --model",
    )

    parser.add_argument(
        "--corrector-provider",
        type=str,
        default=None,
        choices=["openai", "google"],
        help="Provider for corrector. If not set, uses --provider",
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

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum number of concurrent API calls (default: 4)",
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
    judge_model = args.judge_model or args.model
    judge_provider = args.judge_provider or args.provider
    corrector_model = args.corrector_model or args.model
    corrector_provider = args.corrector_provider or args.provider

    print("=" * 60)
    print("WAT MMT DATA CLEANING PIPELINE")
    print("=" * 60)
    print(f"CSV file: {csv_path}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print("\nModel Configuration:")
    print(f"  Judge: {judge_provider}/{judge_model}")
    print(f"  Corrector: {corrector_provider}/{corrector_model}")
    print(f"\nCheckpoint frequency: {args.checkpoint_freq}")
    print(f"Max concurrent tasks: {args.max_concurrent}")
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
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            corrector_model=args.corrector_model,
            corrector_provider=args.corrector_provider,
            checkpoint_frequency=args.checkpoint_freq,
            sample_size=args.sample,
            max_concurrent_tasks=args.max_concurrent,
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
