"""Basic visualization helpers for GUI datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_grid(images: Iterable[Image.Image], path: Path, columns: int = 3) -> None:
    images = list(images)
    if not images:
        raise ValueError("No images provided")

    width, height = images[0].size
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * height))

    for idx, img in enumerate(images):
        col = idx % columns
        row = idx // columns
        grid.paste(img, (col * width, row * height))

    save_image(grid, path)
