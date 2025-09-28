"""Sample NLVR2 examples and optionally save their images."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging

from vlm_datasets.nlvr2 import (
    NLVR2DataConfig,
    find_sample_by_uid,
    load_nlvr2,
    materialize_dataset,
    save_sample_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect NLVR2 samples for inspection")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--uids", nargs="*", default=None, help="Specific sample UIDs to fetch")
    parser.add_argument("--num-samples", type=int, default=4, help="Random samples to draw when UIDs not provided")
    parser.add_argument("--streaming", action="store_true", help="Stream the dataset from HF")
    parser.add_argument("--streaming-take", type=int, default=1024, help="Materialize this many streaming examples")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save images")
    parser.add_argument("--dump-json", type=str, default=None, help="Optional JSONL path for sample metadata")
    return parser.parse_args()


def select_samples(dataset, uids: Sequence[str] | None, num_samples: int, seed: int, limit: int | None):
    if uids:
        samples = []
        for uid in uids:
            samples.append(find_sample_by_uid(dataset, uid, limit=limit))
        return samples

    rng = random.Random(seed)
    materialized = materialize_dataset(dataset, limit)
    indices = list(range(len(materialized)))
    rng.shuffle(indices)
    chosen = indices[: min(num_samples, len(indices))]
    return [materialized[i] for i in chosen]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = NLVR2DataConfig(
        split=args.split,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
        shuffle=args.shuffle,
        streaming_take=args.streaming_take,
    )
    dataset = load_nlvr2(cfg)

    limit = args.streaming_take if args.streaming else None
    samples = select_samples(dataset, args.uids, args.num_samples, args.seed, limit)

    dump_file = None
    if args.dump_json:
        dump_path = Path(args.dump_json)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_file = dump_path.open("w", encoding="utf-8")

    for sample in samples:
        summary = {
            "uid": sample.get("uid"),
            "label": sample.get("label"),
            "sentence": sample.get("sentence"),
            "metadata": sample.get("metadata"),
        }
        print(json.dumps(summary, indent=2))

        if args.save_dir:
            output_dir = Path(args.save_dir)
            saved = save_sample_images(sample, output_dir)
            for path in saved:
                print(f"Saved image: {path}")

        if dump_file:
            dump_file.write(json.dumps(sample) + "\n")

    if dump_file:
        dump_file.close()


if __name__ == "__main__":
    main()
