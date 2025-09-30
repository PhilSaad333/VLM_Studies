"""Helpers for the ScreenSpot-v2 benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from datasets import load_dataset
from PIL import Image, ImageDraw


@dataclass
class ScreenSpotExample:
    image: Image.Image
    bbox_norm: Tuple[float, float, float, float]
    instruction: str
    source: str
    data_type: str

    def bbox_pixels(self) -> Tuple[int, int, int, int]:
        width, height = self.image.size
        x1, y1, x2, y2 = self.bbox_norm
        return (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )

    def draw_bbox(self, color: str = "red", width: int = 4) -> Image.Image:
        img = self.image.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle(self.bbox_pixels(), outline=color, width=width)
        return img


def load_screenspot(split: str = "train", streaming: bool = True, **kwargs) -> Iterable[ScreenSpotExample]:
    dataset = load_dataset(
        "HongxinLi/ScreenSpot_v2",
        split=split,
        streaming=streaming,
        **kwargs,
    )
    for example in dataset:
        yield ScreenSpotExample(
            image=example["image"],
            bbox_norm=tuple(example["bbox"]),
            instruction=example["instruction"],
            source=example.get("data_source", ""),
            data_type=example.get("data_type", ""),
        )
