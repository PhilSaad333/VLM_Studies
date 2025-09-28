"""Helper script to inspect NLVR2 samples by UID or index."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.nlvr2 import (
    NLVR2DataConfig,
    find_sample_by_uid,
    load_nlvr2,
    materialize_dataset,
    save_sample_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect NLVR2 samples")
    parser.add_argument("--uid", type=str, default=None, help="Sample UID to look up")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Dataset index (after materialization) if UID is not provided",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--streaming-take",
        type=int,
        default=512,
        help="Number of examples to materialize when streaming",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of non-streaming examples to load",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the sample's images (PNG format)",
    )
    parser.add_argument(
        "--dump-json",
        type=str,
        default=None,
        help="Optional path to write sample metadata as JSON",
    )
    return parser.parse_args()


def summarize(sample: dict) -> dict:
    metadata = sample.get("metadata", {})
    return {
        "uid": sample.get("uid"),
        "label": sample.get("label"),
        "sentence": sample.get("sentence"),
        "image_info": metadata.get("images"),
        "left_url": metadata.get("left_url"),
        "right_url": metadata.get("right_url"),
    }


def main() -> None:
    args = parse_args()
    cfg = NLVR2DataConfig(
        split=args.split,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
        shuffle=args.shuffle,
        streaming_take=args.streaming_take,
    )
    dataset = load_nlvr2(cfg)

    if args.uid:
        sample = find_sample_by_uid(dataset, args.uid, limit=cfg.streaming_take)
    else:
        if args.index is None:
            raise ValueError("Provide either --uid or --index")
        materialized = materialize_dataset(dataset, args.limit or cfg.streaming_take)
        if args.index < 0 or args.index >= len(materialized):
            raise IndexError(f"Index {args.index} out of range for dataset of length {len(materialized)}")
        sample = materialized[args.index]

    summary = summarize(sample)
    print(json.dumps(summary, indent=2))

    if args.save_dir:
        output_dir = Path(args.save_dir)
        paths = save_sample_images(sample, output_dir)
        for path in paths:
            print(f"Saved image: {path}")

    if args.dump_json:
        dump_path = Path(args.dump_json)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, default=str)
        print(f"Wrote full sample JSON to {dump_path}")


if __name__ == "__main__":
    main()
