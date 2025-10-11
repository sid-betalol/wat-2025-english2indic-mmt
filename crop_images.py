#!/usr/bin/env python3
"""
Script to crop images based on bounding box coordinates from combined datasets.

Usage:
    uv run crop_images.py
    uv run crop_images.py --data_dir data --combined_dir combined_data
    --output_dir cropped_images

This script will:
1. Read the combined CSV files
2. Load images from Malayalam dataset (has one extra example)
3. Crop images based on x, y, width, height coordinates
4. Save cropped images in separate folders for each split
"""

import argparse
import csv
import os

from PIL import Image


def crop_image(image_path, x, y, width, height, output_path):
    """Crop an image based on bounding box coordinates."""
    try:
        with Image.open(image_path) as img:
            # Convert coordinates to integers
            x, y, width, height = int(x), int(y), int(width), int(height)

            # Ensure coordinates are within image bounds
            img_width, img_height = img.size
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            width = max(1, min(width, img_width - x))
            height = max(1, min(height, img_height - y))

            # Crop the image
            cropped = img.crop((x, y, x + width, y + height))

            # Save the cropped image
            cropped.save(output_path)
            return True
    except Exception as e:
        print(f"Error cropping image {image_path}: {e}")
        return False


def crop_images_from_dataset(data_dir, combined_dir, output_dir):
    """Crop images from all combined datasets."""

    splits = ["train", "dev", "test", "challenge"]

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        print(f"\nProcessing {split} split...")

        # Create split-specific output directory
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        # Read combined CSV file
        csv_file = os.path.join(combined_dir, f"combined_{split}.csv")

        if not os.path.exists(csv_file):
            print(f"  CSV file not found: {csv_file}")
            continue

        # Determine image directory (using Malayalam as suggested)
        image_dir = os.path.join(
            data_dir, "malayalam", f"malayalam-visual-genome-{split}.images"
        )

        # Handle challenge set special case
        if split == "challenge":
            image_dir = os.path.join(
                data_dir, "malayalam", "malayalam-visual-genome-chtest.images"
            )

        if not os.path.exists(image_dir):
            print(f"  Image directory not found: {image_dir}")
            continue

        # Process CSV file
        cropped_count = 0
        skipped_count = 0

        with open(csv_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_id = row["image_id"]
                x = row["x"]
                y = row["y"]
                width = row["width"]
                height = row["height"]

                # Construct image path
                image_path = os.path.join(image_dir, f"{image_id}.jpg")

                # Construct output path
                output_path = os.path.join(split_output_dir, f"{image_id}.jpg")

                if os.path.exists(image_path):
                    if crop_image(image_path, x, y, width, height, output_path):
                        cropped_count += 1
                    else:
                        skipped_count += 1
                else:
                    print(f"    Image not found: {image_path}")
                    skipped_count += 1

        print(f"  Cropped {cropped_count} images")
        print(f"  Skipped {skipped_count} images")


def main():
    parser = argparse.ArgumentParser(
        description="Crop images based on bounding box coordinates"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory containing language subdirectories (default: data)",
    )
    parser.add_argument(
        "--combined_dir",
        default="combined_data",
        help="Directory containing combined CSV files (default: combined_data)",
    )
    parser.add_argument(
        "--output_dir",
        default="cropped_images",
        help="Output directory for cropped images (default: cropped_images)",
    )

    args = parser.parse_args()

    print("Cropping images from combined datasets...")
    print(f"Data directory: {args.data_dir}")
    print(f"Combined directory: {args.combined_dir}")
    print(f"Output directory: {args.output_dir}")

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        return

    if not os.path.exists(args.combined_dir):
        print(f"Error: Combined directory '{args.combined_dir}' does not exist")
        return

    crop_images_from_dataset(args.data_dir, args.combined_dir, args.output_dir)
    print("\nImage cropping complete!")


if __name__ == "__main__":
    main()
