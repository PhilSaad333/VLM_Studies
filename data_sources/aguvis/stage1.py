"""Normalized accessors for smolagents/aguvis-stage-1."""
from __future__ import annotations

from typing import Dict, Generator, Iterable, Optional

from datasets import load_dataset
from PIL import Image

STAGE1_CONFIGS = [
    "guienv",
    "omniact",
    "ricoig16k",
    "ricosca",
    "seeclick",
    "ui_refexp",
    "webui350k",
    "widget_captioning",
]


def load_stage1(
    config: str,
    split: str = "train",
    streaming: bool = True,
    **load_kwargs,
) -> Iterable[Dict]:
    """Yield normalized records from AGUVIS stage-1.

    Each record contains:
        image: PIL.Image
        user: instruction string
        assistant: action string (e.g. click/scroll)
        source: original subset name
    """
    if config not in STAGE1_CONFIGS:
        raise ValueError(f"Unknown stage-1 config: {config}")

    dataset = load_dataset(
        "smolagents/aguvis-stage-1",
        config,
        split=split,
        streaming=streaming,
        **load_kwargs,
    )

    for example in dataset:
        image: Image.Image = example["images"][0]
        text_entry = example["texts"][0]
        yield {
            "image": image,
            "user": text_entry.get("user", ""),
            "assistant": text_entry.get("assistant", ""),
            "source": example.get("source", config),
        }
