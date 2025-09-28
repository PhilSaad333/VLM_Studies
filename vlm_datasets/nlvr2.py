"""Utilities for loading and formatting the NLVR2 dataset.

This module centralizes how we access NLVR2, covering both streaming
and cached modes. The goal is to produce examples ready for the
Qwen2-VL chat processor without committing to a specific training loop.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from datasets.exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)

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


def _collect_image_info(example: Dict[str, Any], images: List[Any]) -> List[Dict[str, Any]]:
    info: List[Dict[str, Any]] = []

    for idx, image in enumerate(images):
        slot = None
        if "image" in example and isinstance(example["image"], dict):
            slot = "left" if idx == 0 else "right"
        elif "image0" in example and "image1" in example:
            slot = f"image{idx}"

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

DATASET_SOURCES = [
    ("pingzhili/nlvr2", None),
    ("lmms-lab/NLVR2", None),
    ("HuggingFaceM4/NLVR2", None),
    ("allenai/nlvr2", None),
    ("nlvr", "nlvr2"),
]


def _load_raw_dataset(config: NLVR2DataConfig) -> Dataset | IterableDataset:
    last_error: Exception | None = None
    for path, name in DATASET_SOURCES:
        try:
            kwargs = dict(
                split=config.split,
                streaming=config.streaming,
                data_dir=config.data_dir,
                cache_dir=config.cache_dir,
            )
            if name is not None:
                logger.info("Loading NLVR2 from %s/%s", path, name)
                return load_dataset(path, name, **kwargs)
            logger.info("Loading NLVR2 from %s", path)
            return load_dataset(path, **kwargs)
        except (FileNotFoundError, DatasetNotFoundError, ValueError) as err:
            logger.warning("Failed to load %s (%s): %s", path, name, err)
            last_error = err
            continue
        except Exception as err:  # pragma: no cover - defensive logging
            logger.warning("Unexpected failure loading %s (%s): %s", path, name, err)
            last_error = err
            continue
    if last_error is not None:
        raise last_error
    raise DatasetNotFoundError(
        "Unable to load NLVR2 dataset from the configured sources."
    )


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

    ds = _load_raw_dataset(config)

    if config.streaming and config.shuffle:
        ds = ds.shuffle(buffer_size=config.shuffle_buffer_size, seed=config.seed)
    if config.streaming and config.streaming_take is not None:
        ds = ds.take(config.streaming_take)
    elif not config.streaming and config.shuffle:
        ds = ds.shuffle(seed=config.seed)

    def _format(example: Dict[str, Any]) -> Dict[str, Any]:
        if "image" in example and isinstance(example["image"], dict):
            images: List[Any] = [example["image"]["left"], example["image"]["right"]]
        else:
            images = [example.get("image0"), example.get("image1")]

        if any(image is None for image in images):
            raise ValueError("Example is missing image data")

        uid = example.get("uid") or example.get("identifier") or example.get("id")
        if uid is None:
            uid = f"{config.split}-{hash(example['sentence']) & 0xFFFFFFFF:x}"

        label = example.get("label")
        if isinstance(label, str):
            label_clean = label.strip().lower()
            if label_clean not in {"true", "false"}:
                raise ValueError(f"Unexpected NLVR2 label: {label}")
            label_text = "True" if label_clean == "true" else "False"
        elif isinstance(label, bool):
            label_text = "True" if label else "False"
        else:
            raise ValueError(f"Unsupported label type: {type(label)}")

        prompt = format_nlvr2_prompt(example["sentence"])

        if config.randomize_image_order:
            rng = random.Random(config.seed + hash(uid))
            rng.shuffle(images)

        metadata = {
            "uid": uid,
            "identifier": example.get("identifier"),
            "split": config.split,
            "sentence": example.get("sentence"),
            "label_raw": example.get("label"),
        }

        metadata["images"] = _collect_image_info(example, images)

        return {
            "uid": uid,
            "sentence": example["sentence"].strip(),
            "label": label_text,
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
