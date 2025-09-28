"""LoRA fine-tuning entry points."""
from __future__ import annotations

import logging
from typing import Optional

import torch
from accelerate.utils import set_seed
from peft import LoraConfig
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from vlm_datasets.nlvr2 import NLVR2DataConfig, load_nlvr2, materialize_dataset
from training.config import LoRATrainingConfig
from training.data_collator import Qwen2VLClassificationCollator

logger = logging.getLogger(__name__)


def run_lora_training(config: LoRATrainingConfig) -> SFTTrainer:
    """Fine-tune Qwen2-VL-2B with LoRA on NLVR2."""

    logging.basicConfig(level=logging.INFO)
    set_seed(config.seed)

    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    quantization_config = None
    if config.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=
            getattr(torch, config.qlora_compute_dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    torch_dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)

    model = AutoModelForVision2Seq.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )

    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    train_dataset = load_nlvr2(config.train_data)
    eval_dataset = None
    if config.eval_data is not None:
        eval_dataset = load_nlvr2(config.eval_data)

    train_dataset = materialize_dataset(train_dataset, config.train_data.streaming_take)
    if eval_dataset is not None:
        eval_dataset = materialize_dataset(eval_dataset, config.eval_data.streaming_take)

    collator = Qwen2VLClassificationCollator(processor)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_train_steps if config.max_train_steps is not None else -1,
        logging_dir=config.tensorboard_logdir,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        bf16=config.bf16,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        tokenizer=processor.tokenizer,
        processing_class=processor,
        data_collator=collator,
        max_seq_length=config.max_seq_length,
        packing=False,
    )

    trainer.train()

    trainer.save_model()
    processor.save_pretrained(config.output_dir)

    if config.push_to_hub:
        trainer.push_to_hub(private=config.hub_private_repo, repo_id=config.hub_repo_id)

    return trainer
