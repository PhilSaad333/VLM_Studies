"""Entry script for a small NLVR2 LoRA fine-tuning smoke test."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets.nlvr2 import NLVR2DataConfig
from training.config import LoRATrainingConfig
from training.sft import run_lora_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLVR2 LoRA smoke test")
    parser.add_argument("--output-dir", type=str, default="outputs/smoke_test")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--train-limit", type=int, default=256, help="Number of training examples (streaming take)")
    parser.add_argument("--eval-limit", type=int, default=128, help="Number of eval examples (streaming take)")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--tensorboard-logdir", type=str, default="logs/tb/smoke_test")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--drive-root", type=str, default=None, help="Optional Drive folder (e.g., /content/drive/MyDrive/VLM_Studies_Files)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.drive_root:
        output_dir = Path(args.drive_root) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = NLVR2DataConfig(
        split="train",
        streaming=True,
        streaming_take=args.train_limit,
        cache_dir=args.cache_dir,
        shuffle=True,
    )
    eval_data = NLVR2DataConfig(
        split="validation",
        streaming=True,
        streaming_take=args.eval_limit,
        cache_dir=args.cache_dir,
        shuffle=False,
    )

    config = LoRATrainingConfig(
        model_name=args.model_name,
        output_dir=str(output_dir),
        train_data=train_data,
        eval_data=eval_data,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=args.bf16,
        fp16=args.fp16,
        use_qlora=args.use_qlora,
        tensorboard_logdir=args.tensorboard_logdir,
    )

    run_lora_training(config)


if __name__ == "__main__":
    main()
