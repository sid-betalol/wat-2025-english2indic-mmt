"""Inference script for IndicTrans2 translation."""

import argparse
import logging
import sys
from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class IndicTrans2Translator:
    """Translator using finetuned IndicTrans2 model."""

    # Language code mappings
    LANG_CODE_MAP: ClassVar[dict[str, str]] = {
        "hindi": "hin_Deva",
        "bengali": "ben_Beng",
        "malayalam": "mal_Mlym",
        "odia": "ory_Orya",
        "english": "eng_Latn",
        # Allow direct FLORES codes too
        "hin_deva": "hin_Deva",
        "ben_beng": "ben_Beng",
        "mal_mlym": "mal_Mlym",
        "ory_orya": "ory_Orya",
        "eng_latn": "eng_Latn",
        "hin_Deva": "hin_Deva",
        "ben_Beng": "ben_Beng",
        "mal_Mlym": "mal_Mlym",
        "ory_Orya": "ory_Orya",
        "eng_Latn": "eng_Latn",
    }

    def __init__(
        self,
        model_path: Path | str,
        base_model: str = "ai4bharat/indictrans2-en-indic-dist-200M",
        device: str | None = None,
        is_lora: bool = True,
    ):
        """Initialize the translator.

        Args:
            model_path: Path to the finetuned model
            base_model: Base model name (for LoRA)
            device: Device to use (cuda/mps/cpu), auto-detected if None
            is_lora: Whether the model is a LoRA adapter
        """
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.is_lora = is_lora

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")

        # Load tokenizer with fallback to base model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=True, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {self.model_path}: {e}")
            logger.info(f"Using base model tokenizer: {self.base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model, use_fast=True, trust_remote_code=True
            )

        if self.is_lora:
            # Load base model and LoRA adapter
            logger.info(f"Loading base model: {self.base_model}")
            base = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model, torch_dtype=torch.float32, trust_remote_code=True
            )

            logger.info("Loading LoRA adapter")
            self.model = PeftModel.from_pretrained(base, self.model_path)
            # Merge for faster inference
            self.model = self.model.merge_and_unload()
            # Ensure model is on correct device
            self.model = self.model.to(self.device)
        else:
            # Load full finetuned model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path, torch_dtype=torch.float32, trust_remote_code=True
            )
            # Handle meta tensors by using to_empty() first
            try:
                self.model.to(self.device)
            except RuntimeError as e:
                if "meta tensor" in str(e):
                    logger.info("Model has meta tensors, using to_empty() method")
                    self.model = self.model.to_empty(device=self.device)
                else:
                    raise e
        self.model.eval()

        # Initialize IndicProcessor (use inference=True as per official docs)
        self.processor = IndicProcessor(inference=True)

        logger.info("Model loaded successfully")

    def normalize_language_code(self, lang: str) -> str:
        """Normalize language code to FLORES-200 format.

        Args:
            lang: Language name or code

        Returns:
            FLORES-200 language code
        """
        lang_lower = lang.lower()
        if lang_lower in self.LANG_CODE_MAP:
            return self.LANG_CODE_MAP[lang_lower]
        else:
            raise ValueError(
                f"Unknown language: {lang}. "
                f"Supported: {list(self.LANG_CODE_MAP.keys())}"
            )

    def translate(
        self,
        text: str | list[str],
        target_lang: str,
        source_lang: str = "eng_Latn",
        max_length: int = 256,
        num_beams: int = 5,
        temperature: float = 1.0,
    ) -> str | list[str]:
        """Translate text to target language.

        Args:
            text: Input text or list of texts
            target_lang: Target language (name or FLORES code)
            source_lang: Source language (default: English)
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature

        Returns:
            Translated text or list of translated texts
        """
        # Handle single string vs list
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        # Normalize language codes
        src_code = self.normalize_language_code(source_lang)
        tgt_code = self.normalize_language_code(target_lang)

        # Use IndicProcessor to preprocess inputs (same as training)
        processed_inputs = self.processor.preprocess_batch(
            texts,
            src_lang=src_code,
            tgt_lang=tgt_code,
            is_target=False,  # Match training preprocessing
        )

        # Tokenize
        encoded = self.tokenizer(
            processed_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        # Generate using IndicTrans2 (workaround for beam search bug)
        with torch.no_grad():

            generated_ids = self.model.generate(
                **encoded,
                use_cache=False,
                min_length=0,
                max_length=max_length,
                num_beams=1,
                do_sample=False,
            )

        # Decode using official method
        generated_tokens = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Postprocess using IndicProcessor (as per official docs)
        translations = self.processor.postprocess_batch(generated_tokens, lang=tgt_code)

        # Clean up
        translations = [t.strip() for t in translations]

        return translations[0] if is_single else translations

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str = "eng_Latn",
        batch_size: int = 16,
        **kwargs,
    ) -> list[str]:
        """Translate a batch of texts with batching.

        Args:
            texts: List of input texts
            target_lang: Target language
            source_lang: Source language
            batch_size: Batch size for processing
            **kwargs: Additional arguments for translate()

        Returns:
            List of translated texts
        """
        translations = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_translations = self.translate(
                batch, target_lang, source_lang, **kwargs
            )
            translations.extend(batch_translations)

            if (i + batch_size) % 100 == 0:
                processed = min(i + batch_size, len(texts))
                logger.info(f"Processed {processed}/{len(texts)} texts")

        return translations

    def translate_file(
        self,
        input_file: Path,
        output_file: Path,
        target_lang: str,
        source_lang: str = "eng_Latn",
        text_column: str = "text",
        **kwargs,
    ):
        """Translate texts from a file.

        Args:
            input_file: Input CSV/JSON file
            output_file: Output file path
            target_lang: Target language
            source_lang: Source language
            text_column: Column name containing text
            **kwargs: Additional arguments for translate()
        """
        logger.info(f"Loading texts from {input_file}")

        # Load file
        if input_file.suffix == ".csv":
            df = pd.read_csv(input_file)
        elif input_file.suffix == ".json":
            df = pd.read_json(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_file.suffix}")

        # Get texts
        texts = df[text_column].tolist()

        # Translate
        logger.info(f"Translating {len(texts)} texts to {target_lang}")
        translations = self.translate_batch(texts, target_lang, source_lang, **kwargs)

        # Add translations to dataframe
        df["translation"] = translations

        # Save
        logger.info(f"Saving results to {output_file}")
        if output_file.suffix == ".csv":
            df.to_csv(output_file, index=False)
        elif output_file.suffix == ".json":
            df.to_json(output_file, orient="records", indent=2)

        logger.info("Translation complete!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate text using finetuned IndicTrans2"
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to finetuned model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="ai4bharat/indictrans2-en-indic-dist-200M",
        help="Base model name (for LoRA)",
    )
    parser.add_argument(
        "--is-lora",
        action="store_true",
        default=True,
        help="Whether model is LoRA adapter (default: True)",
    )
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Model is fully finetuned (not LoRA)",
    )

    # Translation mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text to translate")
    group.add_argument("--input-file", type=Path, help="Input file to translate")
    group.add_argument("--interactive", action="store_true", help="Interactive mode")

    # Language settings
    parser.add_argument(
        "--target-lang",
        type=str,
        default="hindi",
        help="Target language (hindi/bengali/malayalam/odia)",
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default="eng_Latn",
        help="Source language",
    )

    # File mode settings
    parser.add_argument("--output-file", type=Path, help="Output file (for file mode)")
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name for text in input file",
    )

    # Generation settings
    parser.add_argument("--max-length", type=int, default=256, help="Maximum length")
    parser.add_argument("--num-beams", type=int, default=5, help="Number of beams")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for file mode",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create translator
    is_lora = not args.full_model
    translator = IndicTrans2Translator(
        model_path=args.model_path,
        base_model=args.base_model,
        is_lora=is_lora,
    )

    try:
        if args.text:
            # Single text mode
            translation = translator.translate(
                args.text,
                args.target_lang,
                args.source_lang,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )
            print(f"\nInput:  {args.text}")
            print(f"Output: {translation}\n")

        elif args.input_file:
            # File mode
            if not args.output_file:
                args.output_file = args.input_file.with_suffix(
                    f".{args.target_lang}{args.input_file.suffix}"
                )

            translator.translate_file(
                input_file=args.input_file,
                output_file=args.output_file,
                target_lang=args.target_lang,
                source_lang=args.source_lang,
                text_column=args.text_column,
                max_length=args.max_length,
                num_beams=args.num_beams,
                batch_size=args.batch_size,
            )

        elif args.interactive:
            # Interactive mode
            print("\n=== Interactive Translation ===")
            print(f"Target language: {args.target_lang}")
            print("Type 'quit' or 'exit' to stop\n")

            while True:
                try:
                    text = input("Enter text: ").strip()
                    if text.lower() in ["quit", "exit", "q"]:
                        break
                    if not text:
                        continue

                    translation = translator.translate(
                        text,
                        args.target_lang,
                        args.source_lang,
                        max_length=args.max_length,
                        num_beams=args.num_beams,
                    )
                    print(f"Translation: {translation}\n")

                except KeyboardInterrupt:
                    print("\nExiting...")
                    break

        return 0

    except Exception as e:
        logger.error(f"Translation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
