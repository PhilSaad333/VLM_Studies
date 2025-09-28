"""Interactive prompt runner for Qwen2-VL models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_datasets.nlvr2 import (
    NLVR2DataConfig,
    find_sample_by_uid,
    load_nlvr2,
    materialize_dataset,
)
from utils.prompts import build_conversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play with Qwen2-VL prompts")
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--uid", type=str, default=None, help="NLVR2 sample UID")
    parser.add_argument("--index", type=int, default=None, help="Dataset index when UID missing")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--streaming-take", type=int, default=1024)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--image", action="append", default=None, help="External image path (can repeat)")
    parser.add_argument("--question", type=str, required=True, help="User question or instruction")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--log-jsonl", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_images_from_paths(paths: List[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        img = Image.open(path)
        images.append(img.convert("RGB"))
    return images


def resolve_dataset_sample(cfg: NLVR2DataConfig, uid: str | None, index: int | None):
    dataset = load_nlvr2(cfg)
    if uid:
        limit = cfg.streaming_take if cfg.streaming else None
        return find_sample_by_uid(dataset, uid, limit=limit)

    if index is None:
        raise ValueError("Provide either --uid or --index when using dataset samples")

    limit = cfg.streaming_take if cfg.streaming else None
    materialized = materialize_dataset(dataset, limit)
    if index < 0 or index >= len(materialized):
        raise IndexError(f"Index {index} out of range (len={len(materialized)})")
    return materialized[index]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    images: List[Image.Image] = []
    metadata = None
    sample_uid = None
    sample_index = args.index

    if args.image:
        images.extend(load_images_from_paths(args.image))
    if args.uid or args.index is not None:
        cfg = NLVR2DataConfig(
            split=args.split,
            streaming=args.streaming,
            cache_dir=args.cache_dir,
            streaming_take=args.streaming_take,
        )
        sample = resolve_dataset_sample(cfg, args.uid, args.index)
        images.extend(sample["images"])
        metadata = sample.get("metadata")
        sample_uid = sample.get("uid")
    if not images:
        raise ValueError("Provide at least one image via --image or NLVR2 sample selection")

    messages = build_conversation(images, args.question, system_prompt=args.system)

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    input_length = inputs["attention_mask"].sum(dim=1)[0]
    generated = output[0, input_length:]
    text = processor.tokenizer.decode(generated, skip_special_tokens=True)

    print("Response:\n" + text)

    if args.log_jsonl:
        log_path = Path(args.log_jsonl)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "question": args.question,
            "system": args.system,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "response": text,
            "metadata": metadata,
            "uid": sample_uid or args.uid,
            "index": sample_index,
            "split": args.split if (args.uid or args.index is not None) else None,
            "images": args.image,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Appended log to {log_path}")


if __name__ == "__main__":
    main()
