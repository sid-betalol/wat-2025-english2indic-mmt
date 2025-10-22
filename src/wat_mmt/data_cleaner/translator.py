"""IndicTrans2 translator wrapper with Apple Silicon (MPS) support."""

from typing import ClassVar

import torch
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class IndicTranslator:
    """Wrapper for IndicTrans2 translation model with MPS support."""

    # Language code mapping
    LANG_CODES: ClassVar[dict[str, str]] = {
        "hindi": "hin_Deva",
        "bengali": "ben_Beng",
        "malayalam": "mal_Mlym",
        "odia": "ory_Orya",
    }

    def __init__(self, model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M"):
        """Initialize the translator.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model: AutoModelForSeq2SeqLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.processor: IndicProcessor | None = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Determine the best available device.

        Returns:
            Device string: 'mps', 'cuda', or 'cpu'
        """

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _lazy_load(self) -> None:
        """Lazy load the model and tokenizer to save memory."""
        if self.model is not None:
            return

        print(f"Loading IndicTrans2 model on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # For MPS, we use float32 as flash_attention_2 is not available
        if self.device == "mps":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            ).to(self.device)
        else:
            # For CUDA, try flash_attention_2 if available
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2",
                ).to(self.device)
            except Exception:
                # Fallback without flash attention
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                ).to(self.device)

        self.processor = IndicProcessor(inference=True)
        print("Model loaded successfully!")

    def translate(self, text: str, target_language: str, max_length: int = 256) -> str:
        """Translate English text to target Indian language.

        Args:
            text: English text to translate
            target_language: Target language name
                (hindi, bengali, malayalam, odia)
            max_length: Maximum length of translation

        Returns:
            Translated text
        """
        self._lazy_load()

        # Get language code
        tgt_lang = self.LANG_CODES.get(target_language.lower())
        if tgt_lang is None:
            raise ValueError(
                f"Unsupported language: {target_language}. "
                f"Supported: {list(self.LANG_CODES.keys())}"
            )

        src_lang = "eng_Latn"

        # Preprocess
        batch = self.processor.preprocess_batch(
            [text], src_lang=src_lang, tgt_lang=tgt_lang
        )

        # Tokenize
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        # Generate translation
        # Disable cache for MPS compatibility
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=False,  # MPS has issues with cache
                min_length=0,
                max_length=max_length,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode
        generated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Postprocess
        translations = self.processor.postprocess_batch(generated_text, lang=tgt_lang)

        return translations[0] if translations else ""

    def translate_batch(
        self, texts: list[str], target_language: str, max_length: int = 256
    ) -> list[str]:
        """Translate a batch of English texts to target Indian language.

        Args:
            texts: List of English texts to translate
            target_language: Target language name
            max_length: Maximum length of translations

        Returns:
            List of translated texts
        """
        self._lazy_load()

        # Get language code
        tgt_lang = self.LANG_CODES.get(target_language.lower())
        if tgt_lang is None:
            raise ValueError(
                f"Unsupported language: {target_language}. "
                f"Supported: {list(self.LANG_CODES.keys())}"
            )

        src_lang = "eng_Latn"

        # Preprocess
        batch = self.processor.preprocess_batch(
            texts, src_lang=src_lang, tgt_lang=tgt_lang
        )

        # Tokenize
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        # Generate translations
        # Disable cache for MPS compatibility
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=False,  # MPS has issues with cache
                min_length=0,
                max_length=max_length,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode
        generated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Postprocess
        translations = self.processor.postprocess_batch(generated_text, lang=tgt_lang)

        return translations
