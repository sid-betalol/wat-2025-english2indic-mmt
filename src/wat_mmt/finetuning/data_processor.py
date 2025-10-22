"""Data processing utilities for IndicTrans2 finetuning."""

import logging
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


class TranslationDataProcessor:
    """Process translation data for IndicTrans2 finetuning."""

    # Mapping from FLORES-200 codes to column prefixes
    LANG_COLUMN_MAP: ClassVar[dict[str, str]] = {
        "hin_Deva": "hindi",
        "ben_Beng": "bengali",
        "mal_Mlym": "malayalam",
        "ory_Orya": "odia",
    }

    def __init__(
        self,
        train_data_path: Path,
        dev_data_path: Path,
        target_languages: list[str],
        source_language: str = "eng_Latn",
        use_corrected: bool = True,
    ):
        """Initialize the data processor.

        Args:
            train_data_path: Path to training data CSV
            dev_data_path: Path to dev/validation data CSV
            target_languages: List of target language codes (FLORES-200)
            source_language: Source language code (default: eng_Latn)
            use_corrected: Whether to use corrected translations
        """
        self.train_data_path = train_data_path
        self.dev_data_path = dev_data_path
        self.target_languages = target_languages
        self.source_language = source_language
        self.use_corrected = use_corrected

    def load_train_data(self) -> pd.DataFrame:
        """Load training data from combined_results.csv."""
        logger.info(f"Loading training data from {self.train_data_path}")
        df = pd.read_csv(self.train_data_path)
        logger.info(f"Loaded {len(df)} training samples")
        return df

    def load_dev_data(self) -> pd.DataFrame:
        """Load dev data from combined_dev.csv."""
        logger.info(f"Loading dev data from {self.dev_data_path}")
        df = pd.read_csv(self.dev_data_path)
        logger.info(f"Loaded {len(df)} dev samples")
        return df

    def get_translation_column(self, lang_code: str) -> str:
        """Get the column name for a given language code.

        Args:
            lang_code: FLORES-200 language code

        Returns:
            Column name for the translation
        """
        lang_name = self.LANG_COLUMN_MAP[lang_code]
        if self.use_corrected:
            return f"{lang_name}_corrected"
        else:
            return f"{lang_name}_original"

    def prepare_training_examples(
        self, df: pd.DataFrame, data_type: Literal["train", "dev"]
    ) -> list[dict]:
        """Convert DataFrame to training examples with language tags.

        Args:
            df: Input DataFrame
            data_type: Either 'train' or 'dev'

        Returns:
            List of training examples with source and target texts
        """
        examples = []

        for _, row in df.iterrows():
            # Get source text
            if data_type == "train":
                source_text = row["english_caption"]
            else:  # dev
                source_text = row["english_text"]

            # Skip if source is missing
            if pd.isna(source_text) or not source_text.strip():
                continue

            # Create one example per target language
            for lang_code in self.target_languages:
                # Get target column name
                if data_type == "train":
                    col_name = self.get_translation_column(lang_code)
                else:  # dev data has different format
                    lang_name = self.LANG_COLUMN_MAP[lang_code]
                    col_name = f"{lang_name}_text"

                # Get target text
                if col_name not in df.columns:
                    logger.warning(
                        f"Column {col_name} not found in {data_type} data"
                    )
                    continue

                target_text = row[col_name]

                # Skip if target is missing
                if pd.isna(target_text) or not target_text.strip():
                    continue

                # Create example with language tags
                example = {
                    "source_text": source_text,
                    "target_text": target_text,
                    "source_lang": self.source_language,
                    "target_lang": lang_code,
                }
                examples.append(example)

        logger.info(
            f"Created {len(examples)} {data_type} examples from {len(df)} rows"
        )
        return examples

    def create_datasets(self) -> DatasetDict:
        """Create HuggingFace DatasetDict for training.

        Returns:
            DatasetDict with 'train' and 'validation' splits
        """
        # Load data
        train_df = self.load_train_data()
        dev_df = self.load_dev_data()

        # Prepare examples
        train_examples = self.prepare_training_examples(train_df, "train")
        dev_examples = self.prepare_training_examples(dev_df, "dev")

        # Create datasets
        train_dataset = Dataset.from_list(train_examples)
        dev_dataset = Dataset.from_list(dev_examples)

        # Create DatasetDict
        dataset_dict = DatasetDict(
            {"train": train_dataset, "validation": dev_dataset}
        )

        logger.info("Dataset statistics:")
        logger.info(f"  Train examples: {len(train_dataset)}")
        logger.info(f"  Dev examples: {len(dev_dataset)}")

        # Print language distribution
        for split_name, dataset in dataset_dict.items():
            lang_counts = {}
            for example in dataset:
                lang = example["target_lang"]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

            logger.info(f"\n{split_name.capitalize()} language distribution:")
            for lang, count in sorted(lang_counts.items()):
                logger.info(f"  {lang}: {count} examples")

        return dataset_dict

    def get_dataset_info(self) -> dict:
        """Get information about the datasets.

        Returns:
            Dictionary with dataset statistics
        """
        train_df = self.load_train_data()
        dev_df = self.load_dev_data()

        train_examples = self.prepare_training_examples(train_df, "train")
        dev_examples = self.prepare_training_examples(dev_df, "dev")

        return {
            "train_rows": len(train_df),
            "dev_rows": len(dev_df),
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "target_languages": self.target_languages,
            "use_corrected": self.use_corrected,
        }
