"""Normalized accessors for smolagents/aguvis-stage-2."""
from __future__ import annotations

from typing import Dict, Iterable

from datasets import load_dataset
from PIL import Image

STAGE2_CONFIGS = [
    "aitw-l1",
    "aitw-l2",
    "aitw-l3",
    "amex-l1",
    "amex-l2",
    "amex-l3",
    "android_control",
    "coat",
    "gui-odyssey-l1",
    "gui-odyssey-l2",
    "gui-odyssey-l3",
    "guiact-web-multi-l1",
    "guiact-web-multi-l2",
    "guiact-web-multi-l3",
    "guiact-web-single",
    "mind2web-l1",
    "mind2web-l2",
    "mind2web-l3",
    "miniwob-l1",
    "miniwob-l2",
    "miniwob-l3",
]


def load_stage2(
    config: str,
    split: str = "train",
    streaming: bool = True,
    **load_kwargs,
) -> Iterable[Dict]:
    """Yield normalized records from AGUVIS stage-2.

    Each record contains:
        image: PIL.Image
        system: system prompt string
        user: user instruction string
        assistant: reasoning + code response (includes <think> and <code>)
        source: original subset name
    """
    if config not in STAGE2_CONFIGS:
        raise ValueError(f"Unknown stage-2 config: {config}")

    dataset = load_dataset(
        "smolagents/aguvis-stage-2",
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
            "system": text_entry.get("system", ""),
            "user": text_entry.get("user", ""),
            "assistant": text_entry.get("assistant", ""),
            "source": example.get("source", config),
        }
