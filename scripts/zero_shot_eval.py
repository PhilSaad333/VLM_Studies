"""CLI wrapper for zero-shot NLVR2 evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.nlvr2 import NLVR2DataConfig
from evals.nlvr2_zero_shot import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot NLVR2 evaluation")
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=str, default="logs/zero_shot_predictions.jsonl")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--streaming-take", type=int, default=None)
    parser.add_argument("--randomize-image-order", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = NLVR2DataConfig(
        split=args.split,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
        shuffle=args.shuffle,
        streaming_take=args.streaming_take or args.limit,
        randomize_image_order=args.randomize_image_order,
    )
    evaluate(
        model_name=args.model_name,
        data_config=data_cfg,
        limit=args.limit,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
