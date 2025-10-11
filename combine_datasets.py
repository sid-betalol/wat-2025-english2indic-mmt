#!/usr/bin/env python3
"""
Script to combine all language datasets (Hindi, Bengali, Malayalam, Odia)
into unified CSV files for each split (train, dev, test, challenge).

Each row represents one image/caption with all target language translations.

Usage:
    uv run combine_datasets.py
    uv run combine_datasets.py --data_dir data --output_dir combined_data

Output:
    - combined_train.csv
    - combined_dev.csv
    - combined_test.csv
    - combined_challenge.csv

Each CSV will have columns:
    image_id, x, y, width, height, english_text,
    hindi_text, bengali_text, malayalam_text, odia_text
"""

import argparse
import csv
import os


def read_dataset_file(file_path):
    """Read a dataset file (txt or tsv) and return as list of rows."""
    try:
        rows = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split("\t")
                    if len(parts) == 7:  # Ensure we have all 7 columns
                        rows.append(parts)
                    else:
                        print(
                            f"Warning: Skipping malformed line in {file_path}: "
                            f"{line}"
                        )
        return rows
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def combine_datasets(data_dir, output_dir):
    """Combine all language datasets into unified CSV files."""

    languages = ["hindi", "bengali", "malayalam", "odia"]
    splits = ["train", "dev", "test", "challenge"]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        print(f"\nProcessing {split} split...")

        # Dictionary to store data by image_id + english_text key
        combined_data = {}

        for lang in languages:
            lang_dir = os.path.join(data_dir, lang)

            # Determine file extension and name based on language and split
            if lang == "odia":
                if split == "challenge":
                    filename = f"{lang}-visual-genome-challenge-test-set.tsv"
                else:
                    filename = f"{lang}-visual-genome-{split}.tsv"
            elif lang == "malayalam":
                if split == "challenge":
                    filename = f"{lang}-visual-genome-chtest.txt"
                else:
                    filename = f"{lang}-visual-genome-{split}.txt"
            else:
                if split == "challenge":
                    filename = f"{lang}-visual-genome-challenge-test-set.txt"
                else:
                    filename = f"{lang}-visual-genome-{split}.txt"

            file_path = os.path.join(lang_dir, filename)

            if os.path.exists(file_path):
                print(f"  Reading {lang} data from {filename}")
                rows = read_dataset_file(file_path)

                if rows is not None:
                    for row in rows:
                        # Create unique key from image_id only
                        key = row[0]  # image_id

                        if key not in combined_data:
                            # Initialize with base data from first occurrence
                            combined_data[key] = {
                                "image_id": row[0],
                                "x": row[1],
                                "y": row[2],
                                "width": row[3],
                                "height": row[4],
                                "english_text": row[5],
                                "hindi_text": "",
                                "bengali_text": "",
                                "malayalam_text": "",
                                "odia_text": "",
                            }

                        # Add translation for this language (overwrites if
                        # multiple entries)
                        combined_data[key][f"{lang}_text"] = row[6]

                    print(f"    Loaded {len(rows)} samples")
                else:
                    print(f"    Failed to load {lang} data")
            else:
                print(f"  File not found: {file_path}")

        if combined_data:
            # Write combined data to CSV
            output_file = os.path.join(output_dir, f"combined_{split}.csv")

            with open(
                output_file, "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(
                    [
                        "image_id",
                        "x",
                        "y",
                        "width",
                        "height",
                        "english_text",
                        "hindi_text",
                        "bengali_text",
                        "malayalam_text",
                        "odia_text",
                    ]
                )

                # Write data rows
                for data in combined_data.values():
                    writer.writerow(
                        [
                            data["image_id"],
                            data["x"],
                            data["y"],
                            data["width"],
                            data["height"],
                            data["english_text"],
                            data["hindi_text"],
                            data["bengali_text"],
                            data["malayalam_text"],
                            data["odia_text"],
                        ]
                    )

            print(
                f"  Saved {len(combined_data)} total samples to {output_file}"
            )

            # Count non-empty translations per language
            lang_counts = {}
            for lang in languages:
                count = sum(
                    1
                    for data in combined_data.values()
                    if data[f"{lang}_text"].strip()
                )
                lang_counts[lang] = count

            print(f"  Language distribution: {lang_counts}")
        else:
            print(f"  No data found for {split} split")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multilingual Visual Genome datasets"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory containing language subdirectories (default: data)",
    )
    parser.add_argument(
        "--output_dir",
        default="combined_data",
        help="Output directory for combined CSV files (default: combined_data)",
    )

    args = parser.parse_args()

    print("Combining Visual Genome datasets...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        return

    combine_datasets(args.data_dir, args.output_dir)
    print("\nDataset combination complete!")


if __name__ == "__main__":
    main()
