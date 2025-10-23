"""Main finetuning script for IndicTrans2."""

import argparse
import logging
import sys
from pathlib import Path
import os

import evaluate
import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from IndicTransToolkit.processor import IndicProcessor

from .config import FinetuningConfig
from .data_processor import TranslationDataProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class IndicTrans2Finetuner:
    """Finetuner for IndicTrans2 model."""

    def __init__(self, config: FinetuningConfig):
        """Initialize the finetuner.

        Args:
            config: Finetuning configuration
        """
        self.config = config
        set_seed(config.seed)

        # Set device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.bleu_metric = evaluate.load("sacrebleu")

    def load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            use_fast=True,
            trust_remote_code=True,
        )

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            torch_dtype=torch.float32,  # Use float32 for Mac M2
            trust_remote_code=True,
        )

        # Initialize IndicProcessor
        self.processor = IndicProcessor(inference=False)

        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")

        # Apply LoRA if specified
        if self.config.method == "lora":
            self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA to the model."""
        logger.info("Applying LoRA configuration")

        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def preprocess_function(self, examples):
        """Preprocess examples for training using IndicProcessor.

        Args:
            examples: Batch of examples

        Returns:
            Tokenized inputs and labels
        """
        # Prepare source and target texts
        source_texts = examples["source_text"]
        target_texts = examples["target_text"]
        source_langs = examples["source_lang"]
        target_langs = examples["target_lang"]

        # Use IndicProcessor to preprocess inputs (with language tags)
        processed_inputs = self.processor.preprocess_batch(
            source_texts,
            src_lang=source_langs[0],  # All should be same source lang
            tgt_lang=target_langs[0],  # All should be same target lang
        )

        # Use IndicProcessor to preprocess targets (without language tags)
        processed_targets = self.processor.preprocess_batch(
            target_texts,
            src_lang=target_langs[0],  # Target becomes source for labels
            tgt_lang=target_langs[0],  # Same language
        )

        # Tokenize inputs
        model_inputs = self.tokenizer(
            processed_inputs,
            max_length=self.config.max_length,
            truncation=True,
            padding=False,
        )

        # Tokenize targets
        labels = self.tokenizer(
            processed_targets,
            max_length=self.config.max_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_preds):
        """Compute evaluation metrics.

        Args:
            eval_preds: Predictions and labels

        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_preds

        # Decode predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Replace -100 in labels (used for padding)
        labels = [
            [label_id for label_id in label if label_id != -100] for label in labels
        ]

        # Decode
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean up predictions and labels
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        # Compute BLEU
        try:
            result = self.bleu_metric.compute(
                predictions=decoded_preds, references=decoded_labels
            )
            bleu_score = result["score"]
        except Exception as e:
            logger.warning(f"Error computing BLEU: {e}")
            bleu_score = 0.0

        return {"bleu": bleu_score}

    def train(self):
        """Run the finetuning process."""
        logger.info("Starting finetuning process")

        # Initialize distributed training if multi-GPU
        if self.config.use_multi_gpu:
            if not dist.is_initialized():
                dist.init_process_group(backend=self.config.ddp_backend)
            logger.info(
                f"Initialized distributed training with {dist.get_world_size()} processes"
            )

        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Prepare data
        logger.info("Preparing datasets")
        data_processor = TranslationDataProcessor(
            train_data_path=self.config.train_data_path,
            dev_data_path=self.config.dev_data_path,
            target_languages=self.config.target_languages,
            source_language=self.config.source_language,
            use_corrected=self.config.use_corrected,
        )

        dataset_dict = data_processor.create_datasets()

        # Preprocess datasets
        logger.info("Tokenizing datasets")
        tokenized_datasets = dataset_dict.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, padding=True
        )

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps if self.config.max_steps > 0 else -1,
            per_device_train_batch_size=(self.config.per_device_train_batch_size),
            per_device_eval_batch_size=(self.config.per_device_eval_batch_size),
            gradient_accumulation_steps=(self.config.gradient_accumulation_steps),
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            predict_with_generate=True,
            generation_max_length=self.config.max_length,
            generation_num_beams=self.config.num_beams,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            seed=self.config.seed,
            push_to_hub=False,
            # Multi-GPU settings
            ddp_backend=(
                self.config.ddp_backend if self.config.use_multi_gpu else None
            ),
            ddp_find_unused_parameters=(
                self.config.ddp_find_unused_parameters
                if self.config.use_multi_gpu
                else None
            ),
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Train
        logger.info("Starting training")
        train_result = trainer.train()

        # Save final model
        logger.info(f"Saving model to {self.config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Final evaluation
        logger.info("Running final evaluation")
        try:
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
            logger.info(f"Final BLEU score: {eval_metrics['eval_bleu']:.2f}")
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            logger.info("Continuing without evaluation metrics...")
            eval_metrics = {"eval_bleu": 0.0}

        logger.info("Finetuning complete!")

        return trainer, eval_metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune IndicTrans2 for English to Indic translation"
    )

    # Data arguments
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("combined_processed_data/combined_results.csv"),
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--dev-data",
        type=Path,
        default=Path("combined_data/combined_dev.csv"),
        help="Path to dev data CSV",
    )
    parser.add_argument(
        "--use-corrected",
        action="store_true",
        default=True,
        help="Use corrected translations (default: True)",
    )
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="Use original translations instead of corrected",
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="ai4bharat/indictrans2-en-indic-dist-200M",
        help="Model name or path",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Finetuning method: lora or full",
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/indictrans2-finetuned"),
        help="Output directory for saved models",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum training steps (overrides epochs)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-5, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")

    # Multi-GPU arguments
    parser.add_argument(
        "--multi-gpu", action="store_true", help="Enable multi-GPU training"
    )
    parser.add_argument(
        "--ddp-backend", type=str, default="nccl", help="DDP backend (nccl/gloo)"
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action="store_true",
        help="Find unused parameters in DDP",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create config
    config = FinetuningConfig(
        model_name=args.model_name,
        method=args.method,
        train_data_path=args.train_data,
        dev_data_path=args.dev_data,
        use_corrected=not args.use_original,  # Invert the flag
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        # Multi-GPU settings
        use_multi_gpu=args.multi_gpu,
        ddp_backend=args.ddp_backend,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )

    # Update LoRA config if using LoRA
    if config.method == "lora":
        config.lora.r = args.lora_r
        config.lora.lora_alpha = args.lora_alpha
        config.lora.lora_dropout = args.lora_dropout

    # Log configuration
    logger.info("=" * 80)
    logger.info("Finetuning Configuration")
    logger.info("=" * 80)
    logger.info(f"Method: {config.method}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Train data: {config.train_data_path}")
    logger.info(f"Dev data: {config.dev_data_path}")
    logger.info(f"Use corrected: {config.use_corrected}")
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    logger.info(f"Batch size: {config.per_device_train_batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    if config.method == "lora":
        logger.info(f"LoRA rank: {config.lora.r}")
        logger.info(f"LoRA alpha: {config.lora.lora_alpha}")
    if config.use_multi_gpu:
        logger.info("Multi-GPU: Enabled")
        logger.info(f"DDP backend: {config.ddp_backend}")
        logger.info(f"DDP find unused parameters: {config.ddp_find_unused_parameters}")
    logger.info("=" * 80)

    # Create finetuner
    finetuner = IndicTrans2Finetuner(config)

    # Run training
    try:
        finetuner.train()
        logger.info("Training completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
