"""Custom data collators for Qwen2-VL fine-tuning."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import AutoProcessor

from datasets.nlvr2 import build_chat_messages
from utils.prompts import attach_assistant_response


class Qwen2VLClassificationCollator:
    """Prepare NLVR2 batches for causal LM training.

    Each item in the incoming batch is expected to contain:
        - ``images``: list with two PIL images
        - ``sentence``: NLVR2 statement
        - ``label``: "True" or "False"

    The collator builds a chat-style conversation and leverages the Qwen2-VL
    processor to obtain tokenized inputs, masked labels, and image tensors.
    """

    def __init__(
        self,
        processor: AutoProcessor,
        pad_to_multiple_of: int | None = 8,
    ) -> None:
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids, attention_masks, labels, pixel_values = [], [], [], []

        for example in batch:
            messages = build_chat_messages(example["sentence"], example["images"])
            full_messages = attach_assistant_response(messages, example["label"])

            processed = self.processor.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            prompt_only = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            ids = processed["input_ids"][0]
            mask = processed["attention_mask"][0]
            pixels = processed["pixel_values"][0]

            labels_tensor = ids.clone()
            prompt_len = prompt_only["input_ids"].shape[-1]
            labels_tensor[:prompt_len] = -100

            input_ids.append(ids)
            attention_masks.append(mask)
            labels.append(labels_tensor)
            pixel_values.append(pixels)

        batch_dict = self.processor.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels,
            },
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        pixel_values = torch.stack(pixel_values, dim=0)
        batch_dict["pixel_values"] = pixel_values
        return batch_dict
