"""Configuration objects for training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from datasets import Dataset, IterableDataset

from vlm_datasets.nlvr2 import NLVR2DataConfig


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA fine-tuning on NLVR2."""

    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir: str = "outputs/nlvr2_lora"

    train_data: NLVR2DataConfig = field(
        default_factory=lambda: NLVR2DataConfig(
            split="train", streaming=True, streaming_take=1024
        )
    )
    eval_data: Optional[NLVR2DataConfig] = field(
        default_factory=lambda: NLVR2DataConfig(
            split="validation", streaming=True, streaming_take=256
        )
    )

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    num_train_epochs: float = 1.0
    max_train_steps: Optional[int] = None
    max_seq_length: int = 2048

    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 2

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_qlora: bool = False
    qlora_compute_dtype: str = "bfloat16"

    tensorboard_logdir: str = "logs/tb"
    push_to_hub: bool = False
    hub_private_repo: bool = True
    hub_repo_id: Optional[str] = None


@dataclass
class ZeroShotEvalConfig:
    """Configuration for zero-shot NLVR2 evaluation."""

    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    split: str = "validation"
    limit: Optional[int] = 512
    batch_size: int = 4
    streaming: bool = True
    cache_dir: Optional[str] = None
    seed: int = 42
    output_path: Optional[str] = "logs/zero_shot_predictions.jsonl"

    data_config: NLVR2DataConfig = field(
        default_factory=lambda: NLVR2DataConfig(
            split="validation", streaming=True, streaming_take=512
        )
    )
