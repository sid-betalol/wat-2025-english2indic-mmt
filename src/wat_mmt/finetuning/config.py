"""Configuration classes for IndicTrans2 finetuning."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class LoRAConfig:
    """LoRA-specific configuration parameters."""

    r: int = 16  # Rank of the low-rank matrices
    lora_alpha: int = 32  # Scaling factor (typically 2*r)
    lora_dropout: float = 0.1  # Dropout probability
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # Bias handling: 'none', 'all', or 'lora_only'
    task_type: str = "SEQ_2_SEQ_LM"  # Task type for PEFT


@dataclass
class FinetuningConfig:
    """Main configuration for finetuning."""

    # Model configuration
    model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M"
    cache_dir: str | None = None

    # Training method
    method: Literal["lora", "full"] = "lora"

    # Data configuration
    train_data_path: Path = field(
        default_factory=lambda: Path("combined_processed_data/combined_results.csv")
    )
    dev_data_path: Path = field(
        default_factory=lambda: Path("combined_data/combined_dev.csv")
    )
    use_corrected: bool = True  # Use corrected vs original translations

    # Target languages (FLORES-200 codes)
    target_languages: list[str] = field(
        default_factory=lambda: [
            "hin_Deva",  # Hindi
            "ben_Beng",  # Bengali
            "mal_Mlym",  # Malayalam
            "ory_Orya",  # Odia
        ]
    )
    source_language: str = "eng_Latn"  # English

    # Training hyperparameters
    output_dir: Path = field(
        default_factory=lambda: Path("models/indictrans2-finetuned")
    )
    num_train_epochs: int = 3
    max_steps: int = -1  # Override epochs if > 0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Optimization
    fp16: bool = False  # Use mixed precision (set to False for Mac M2)
    bf16: bool = False  # Use bfloat16 (Mac M2 doesn't support this well)
    gradient_checkpointing: bool = False

    # Multi-GPU settings
    use_multi_gpu: bool = False  # Enable multi-GPU training
    ddp_backend: str = "nccl"  # Distributed training backend (nccl for CUDA)
    ddp_find_unused_parameters: bool = False  # Find unused parameters in DDP

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100

    # Generation parameters for evaluation
    max_length: int = 256
    num_beams: int = 5

    # LoRA configuration (only used if method='lora')
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Misc
    seed: int = 42
    report_to: str = "none"  # Can be 'wandb', 'tensorboard', etc.

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.train_data_path, str):
            self.train_data_path = Path(self.train_data_path)
        if isinstance(self.dev_data_path, str):
            self.dev_data_path = Path(self.dev_data_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_language_name(self, flores_code: str) -> str:
        """Convert FLORES-200 code to language name."""
        mapping = {
            "hin_Deva": "hindi",
            "ben_Beng": "bengali",
            "mal_Mlym": "malayalam",
            "ory_Orya": "odia",
            "eng_Latn": "english",
        }
        return mapping.get(flores_code, flores_code)
