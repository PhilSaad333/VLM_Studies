"""Utilities for surprisal analysis and sample inspection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from data_sources.aguvis.stage1 import STAGE1_CONFIGS, load_stage1
from data_sources.aguvis.stage2 import STAGE2_CONFIGS, load_stage2


_DATASET_LOADERS = {
    "stage1": (load_stage1, STAGE1_CONFIGS),
    "stage2": (load_stage2, STAGE2_CONFIGS),
}


@dataclass
class SampleLocator:
    dataset: str
    config: str
    split: str
    index: int

    @property
    def sample_id(self) -> str:
        return make_sample_id(self.dataset, self.config, self.split, self.index)


def make_sample_id(dataset: str, config: str, split: str, index: int) -> str:
    """Format a stable sample identifier."""
    return f"{dataset}:{config}:{split}:{index:08d}"


def parse_sample_id(sample_id: str) -> SampleLocator:
    """Parse a sample id emitted by the surprisal evaluator."""
    parts = sample_id.split(":")
    if len(parts) != 4:
        raise ValueError(f"Unexpected sample_id format: {sample_id}")
    dataset, config, split, index_str = parts
    try:
        index = int(index_str)
    except ValueError as exc:
        raise ValueError(f"Invalid index in sample_id: {sample_id}") from exc
    return SampleLocator(dataset=dataset, config=config, split=split, index=index)


def _resolve_loader(dataset: str, config: str):
    if dataset not in _DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset}")
    loader_fn, configs = _DATASET_LOADERS[dataset]
    if config not in configs:
        raise ValueError(f"Config '{config}' not valid for {dataset}. Available: {sorted(configs)}")
    return loader_fn


def fetch_sample(sample_id: str, streaming: bool = True):
    """Return the dataset record for a given sample id."""
    locator = parse_sample_id(sample_id)
    loader_fn = _resolve_loader(locator.dataset, locator.config)
    iterator: Iterable[Dict] = loader_fn(locator.config, split=locator.split, streaming=streaming)
    for idx, sample in enumerate(iterator):
        if idx == locator.index:
            return sample
    raise IndexError(f"Sample index {locator.index} not found for {sample_id}")


def show_sample(sample_id: str, streaming: bool = True, show_image: bool = True):
    """Fetch and optionally display a dataset sample for quick inspection."""
    sample = fetch_sample(sample_id, streaming=streaming)
    info = {"sample_id": sample_id}
    for key in ("system", "user", "assistant", "source", "data_type"):
        if key in sample:
            info[key] = sample[key]
    image = sample.get("image")
    info["image_size"] = getattr(image, "size", None)
    if show_image and image is not None:
        try:
            from IPython.display import display
        except ImportError:
            display = None
        if display is not None:
            display(image)
    return info
