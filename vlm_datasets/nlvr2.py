"""Utilities for loading and formatting the NLVR2 dataset.

This module centralizes how we access NLVR2, covering both streaming
and cached modes. The goal is to produce examples ready for the
Qwen2-VL chat processor without committing to a specific training loop.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset

from utils.prompts import build_conversation, format_nlvr2_prompt


@dataclass
class NLVR2DataConfig:
    """Configuration for NLVR2 loading.

    Attributes:
        split: Which split to load (e.g., "train", "validation", "test").
        streaming: Whether to stream examples from the hub instead of
            downloading the full archive.
        data_dir: Optional local directory containing an extracted copy of
            NLVR2. Leave as ``None`` to rely on the Hugging Face mirrors.
        cache_dir: Optional datasets cache directory. Useful when running on
            Colab with Drive mounted.
        shuffle: Shuffle the dataset (only supported when ``streaming`` is
            ``True`` or when the dataset is materialized locally).
        shuffle_buffer_size: Buffer size used for streaming shuffle.
        seed: Seed for shuffle and optional image order randomization.
        randomize_image_order: Whether to randomly swap left/right images per
            example. Helpful for order-sensitivity ablations.
    """

    split: str = "train"
    streaming: bool = True
    data_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    shuffle: bool = True
    shuffle_buffer_size: int = 10_000
    seed: int = 42
    randomize_image_order: bool = False
    streaming_take: Optional[int] = None


def _collect_image_info(image_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    info: List[Dict[str, Any]] = []

    for slot in ("left", "right"):
        image = image_dict.get(slot)
        if image is None:
            continue
        width = getattr(image, "width", None)
        height = getattr(image, "height", None)
        filename = getattr(image, "filename", None)
        info.append(
            {
                "slot": slot,
                "width": width,
                "height": height,
                "filename": filename,
            }
        )
    return info


def load_nlvr2(config: NLVR2DataConfig) -> Dataset | IterableDataset:
    """Load NLVR2 according to ``config`` and normalize its structure.

    Returns a dataset where each example contains:
        - ``uid``: Unique identifier.
        - ``sentence``: Natural language statement.
        - ``label``: "True" or "False" (string).
        - ``images``: List with two PIL images (left, right order).
        - ``prompt``: Text prompt injected into the model.

    The dataset retains streaming semantics when ``config.streaming`` is True.
    """

    ds = load_dataset(
        "nlvr",
        "nlvr2",
        split=config.split,
        streaming=config.streaming,
        data_dir=config.data_dir,
        cache_dir=config.cache_dir,
    )

    if config.streaming and config.shuffle:
        ds = ds.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.seed)
    if config.streaming and config.streaming_take is not None:
        ds = ds.take(config.streaming_take)
    elif not config.streaming and config.shuffle:
        ds = ds.shuffle(seed=config.seed)

    def _format(example: Dict[str, Any]) -> Dict[str, Any]:
        images: List[Any] = [example["image"]["left"], example["image"]["right"]]
        if config.randomize_image_order:
            rng = random.Random(config.seed + hash(example["uid"]))
            rng.shuffle(images)

        label = example["label"].strip()
        if label.lower() not in {"true", "false"}:
            raise ValueError(f"Unexpected NLVR2 label: {label}")

        prompt = format_nlvr2_prompt(example["sentence"])

        metadata = {
            "uid": example.get("uid"),
            "identifier": example.get("identifier"),
            "pair_id": example.get("pair_id"),
            "split": config.split,
            "left_url": example.get("left_url"),
            "right_url": example.get("right_url"),
            "directory": example.get("directory"),
            "filenames": example.get("filenames"),
            "sentence": example.get("sentence"),
        }

        metadata["images"] = _collect_image_info(example["image"])

        return {
            "uid": example["uid"],
            "sentence": example["sentence"].strip(),
            "label": "True" if label.lower() == "true" else "False",
            "images": images,
            "prompt": prompt,
            "metadata": metadata,
        }

    # ``remove_columns`` is not supported for iterable datasets; mapping will
    # keep the original fields around, which is harmless for our purposes.
    ds = ds.map(_format)
    return ds


def get_nlvr2_splits(
    train_cfg: NLVR2DataConfig,
    dev_cfg: Optional[NLVR2DataConfig] = None,
) -> DatasetDict:
    """Convenience loader that returns a ``DatasetDict`` with train/dev splits."""

    dataset_dict: Dict[str, Dataset | IterableDataset] = {
        train_cfg.split: load_nlvr2(train_cfg)
    }
    if dev_cfg is not None:
        dataset_dict[dev_cfg.split] = load_nlvr2(dev_cfg)
    return DatasetDict(dataset_dict)


def build_chat_messages(sentence: str, images: List[Any], system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return NLVR2 classification messages for Qwen-style processors."""

    prompt = format_nlvr2_prompt(sentence)
    return build_conversation(images, prompt, system_prompt)


def materialize_dataset(dataset: Dataset | IterableDataset, limit: Optional[int]) -> Dataset:
    """Return a non-streaming Dataset limited to ``limit`` examples.

    Parameters
    ----------
    dataset: Dataset | IterableDataset
        Hugging Face dataset object to materialize.
    limit: Optional[int]
        Number of examples to keep when converting from streaming. Required for
        iterable datasets.
    """

    if isinstance(dataset, IterableDataset):
        if limit is None:
            raise ValueError(
                "Materializing a streaming dataset requires a limit. "
                "Set `streaming_take` in NLVR2DataConfig."
            )
        return Dataset.from_list(list(islice(dataset, limit)))

    if limit is not None:
        limit = min(limit, len(dataset))
        return dataset.select(range(limit))
    return dataset


def find_sample_by_uid(dataset: Dataset | IterableDataset, uid: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Locate a single NLVR2 example by UID.

    ``limit`` is required when ``dataset`` is streaming so we can materialize a
    finite prefix.
    """

    materialized = materialize_dataset(dataset, limit)
    for example in materialized:
        if example.get("uid") == uid:
            return example
    raise KeyError(f"UID {uid} not found in provided dataset slice")


def save_sample_images(example: Dict[str, Any], output_dir: Path) -> List[Path]:
    """Persist the left/right images from an example for later inspection."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    metadata_images = example.get("metadata", {}).get("images", [])

    for idx, image in enumerate(example.get("images", [])):
        slot = metadata_images[idx].get("slot") if idx < len(metadata_images) else f"image_{idx}"
        filename = metadata_images[idx].get("filename") if idx < len(metadata_images) else None
        if not filename:
            filename = f"{example.get('uid', 'sample')}_{slot}.png"
        path = output_dir / filename
        image.save(path)
        saved_paths.append(path)

    return saved_paths
