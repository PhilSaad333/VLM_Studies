"""Compute token-level surprisal statistics for multimodal datasets."""
from __future__ import annotations

import argparse
import json
import math
import time
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from data_sources.aguvis.stage1 import STAGE1_CONFIGS, load_stage1
from data_sources.aguvis.stage2 import STAGE2_CONFIGS, load_stage2
from evals.surprisal_utils import make_sample_id

LOG_2_INV = 1.0 / math.log(2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset surprisal measurement for SmolVLM2.")
    parser.add_argument("--dataset", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--config", required=True, help="Dataset configuration name (e.g. webui350k).")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--limit", type=int, help="Maximum number of samples to evaluate.")
    parser.add_argument("--skip", type=int, default=0, help="Number of samples to skip from the start.")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/surprisal"))
    parser.add_argument("--device", default=None, help="Torch device override (e.g. cuda, cuda:1).")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype (float16, bfloat16, float32).")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bar.")
    parser.add_argument("--lora-ranks", type=int, nargs="*", default=[8, 16, 32, 64])
    parser.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated suffixes of Linear modules to include when estimating LoRA params.",
    )
    parser.add_argument(
        "--lora-name-contains",
        default="model.text_model",
        help="Comma-separated substrings that module names must contain when counting LoRA params."
             " Leave empty to include all modules.",
    )
    parser.add_argument(
        "--log-token-details",
        action="store_true",
        help="Include per-token surprisal data in the JSONL output (larger files).",
    )
    parser.add_argument("--description", help="Optional note stored in the metadata JSON.")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Attribute torch.{dtype_name} is not a dtype")
    return dtype


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_config(dataset: str, config: str) -> None:
    valid = STAGE1_CONFIGS if dataset == "stage1" else STAGE2_CONFIGS
    if config not in valid:
        raise ValueError(f"Config '{config}' not valid for {dataset}. Choices: {sorted(valid)}")


def get_loader(dataset: str):
    return load_stage1 if dataset == "stage1" else load_stage2


def prepare_iterator(
    dataset: str,
    config: str,
    split: str,
    skip: int,
    limit: int | None,
) -> Iterator[Dict]:
    loader = get_loader(dataset)
    iterator: Iterable[Dict] = loader(config, split=split, streaming=True)
    if skip:
        iterator = islice(iterator, skip, None)
    if limit is not None:
        iterator = islice(iterator, limit)
    return iterator


def build_messages(dataset: str, sample: Dict) -> List[Dict]:
    image = sample.get("image")
    if dataset == "stage1":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample.get("user", "")},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample.get("assistant", "")}],
            },
        ]
    system = sample.get("system")
    messages: List[Dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample.get("user", "")},
            ],
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample.get("assistant", "")}],
        }
    )
    return messages


def gather_image_token_ids(tokenizer) -> Sequence[int]:
    ids: List[int] = []
    special = [
        "<image>",
        "<fake_token_around_image>",
        "<global-img>",
    ]
    for token in special:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != -1:
            ids.append(int(token_id))
    return tuple(sorted(set(ids)))


def compute_token_stats(
    model,
    inputs,
    tokenizer,
    image_token_ids: Sequence[int],
    include_token_details: bool,
) -> Dict:
    labels = inputs["input_ids"].clone()
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    labels[:, 0] = -100
    batch = {key: value for key, value in inputs.items()}
    batch["labels"] = labels
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits.float()
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    shifted_log_probs = log_probs[:, :-1, :]
    target_ids = batch["input_ids"][:, 1:]
    target_labels = batch["labels"][:, 1:]
    gather = torch.gather(shifted_log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = target_labels != -100
    masked_log_probs = gather[mask]
    token_ids = target_ids[mask]

    surprisal_nats = (-masked_log_probs).cpu()
    token_ids_cpu = token_ids.cpu()
    surprisal_bits = surprisal_nats * LOG_2_INV

    num_tokens = surprisal_bits.numel()
    if num_tokens == 0:
        return {
            "num_tokens": 0,
            "num_image_tokens": 0,
            "num_text_tokens": 0,
            "total_surprisal_bits": 0.0,
            "total_surprisal_nats": 0.0,
            "avg_surprisal_bits": 0.0,
            "avg_surprisal_nats": 0.0,
            "image_surprisal_bits": 0.0,
            "text_surprisal_bits": 0.0,
            "token_details": [] if include_token_details else None,
        }

    image_token_set = set(int(tid) for tid in image_token_ids)
    token_ids_list = token_ids_cpu.tolist()
    bits_list = surprisal_bits.tolist()

    image_bits = 0.0
    text_bits = 0.0
    token_details = []
    for token_id, bits in zip(token_ids_list, bits_list):
        if token_id in image_token_set:
            image_bits += bits
        else:
            text_bits += bits
        if include_token_details:
            token_details.append(
                {
                    "token_id": int(token_id),
                    "token": tokenizer.convert_ids_to_tokens([token_id])[0],
                    "surprisal_bits": float(bits),
                }
            )

    total_bits = image_bits + text_bits
    total_nats = float(total_bits / LOG_2_INV)
    avg_bits = total_bits / num_tokens
    avg_nats = float(avg_bits / LOG_2_INV)
    return {
        "num_tokens": int(num_tokens),
        "num_image_tokens": int(sum(1 for token_id in token_ids_list if token_id in image_token_set)),
        "num_text_tokens": int(sum(1 for token_id in token_ids_list if token_id not in image_token_set)),
        "total_surprisal_bits": float(total_bits),
        "total_surprisal_nats": float(total_nats),
        "avg_surprisal_bits": float(avg_bits),
        "avg_surprisal_nats": float(avg_nats),
        "image_surprisal_bits": float(image_bits),
        "text_surprisal_bits": float(text_bits),
        "token_details": token_details if include_token_details else None,
    }

def estimate_lora_params(model, target_suffixes: Sequence[str], name_filters: Sequence[str], rank: int) -> int:
    if not target_suffixes:
        return 0
    total = 0
    suffixes = tuple(target_suffixes)
    filters = tuple(name_filters)
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if suffixes and not any(name.endswith(suffix) for suffix in suffixes):
            continue
        if filters and not any(fragment in name for fragment in filters):
            continue
        out_features, in_features = module.weight.shape
        total += rank * (in_features + out_features)
    return total


def main() -> None:
    args = parse_args()
    validate_config(args.dataset, args.config)

    dtype = resolve_dtype(args.dtype)
    device = resolve_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    tokenizer = processor.tokenizer
    image_token_ids = gather_image_token_ids(tokenizer)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.dataset}_{args.config}_{args.split}_{timestamp}"
    jsonl_path = args.output_dir / f"{prefix}.jsonl"
    meta_path = args.output_dir / f"{prefix}_meta.json"

    iterator = prepare_iterator(args.dataset, args.config, args.split, args.skip, args.limit)
    progress_iter = tqdm(iterator, desc="surprisal", unit="sample") if args.progress else iterator

    aggregates = {
        "samples": 0,
        "tokens": 0,
        "bits": 0.0,
        "image_bits": 0.0,
        "text_bits": 0.0,
    }

    output_records: List[Dict] = []

    for local_idx, sample in enumerate(progress_iter):
        global_index = args.skip + local_idx
        sample_id = make_sample_id(args.dataset, args.config, args.split, global_index)
        messages = build_messages(args.dataset, sample)

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=dtype)

        stats = compute_token_stats(model, inputs, tokenizer, image_token_ids, args.log_token_details)

        record = {
            "sample_id": sample_id,
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "index": global_index,
            "source": sample.get("source"),
            "total_surprisal_bits": stats["total_surprisal_bits"],
            "total_surprisal_nats": stats["total_surprisal_nats"],
            "avg_surprisal_bits": stats["avg_surprisal_bits"],
            "avg_surprisal_nats": stats["avg_surprisal_nats"],
            "num_tokens": stats["num_tokens"],
            "num_image_tokens": stats["num_image_tokens"],
            "num_text_tokens": stats["num_text_tokens"],
            "image_surprisal_bits": stats["image_surprisal_bits"],
            "text_surprisal_bits": stats["text_surprisal_bits"],
            "assistant_len": len(sample.get("assistant", "")),
        }
        if args.dataset == "stage2":
            record["system_len"] = len(sample.get("system", ""))
        if args.log_token_details and stats["token_details"] is not None:
            record["token_details"] = stats["token_details"]

        aggregates["samples"] += 1
        aggregates["tokens"] += stats["num_tokens"]
        aggregates["bits"] += stats["total_surprisal_bits"]
        aggregates["image_bits"] += stats["image_surprisal_bits"]
        aggregates["text_bits"] += stats["text_surprisal_bits"]

        output_records.append(record)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")

    lora_targets = [frag.strip() for frag in args.lora_targets.split(",") if frag.strip()]
    name_filters = [frag.strip() for frag in args.lora_name_contains.split(",") if frag.strip()]
    lora_counts = {
        str(rank): int(estimate_lora_params(model, lora_targets, name_filters, rank))
        for rank in args.lora_ranks
    }

    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "limit": args.limit,
        "skip": args.skip,
        "samples": aggregates["samples"],
        "total_tokens": aggregates["tokens"],
        "total_surprisal_bits": aggregates["bits"],
        "total_surprisal_nats": aggregates["bits"] / LOG_2_INV if aggregates["bits"] else 0.0,
        "mean_bits_per_token": (aggregates["bits"] / aggregates["tokens"]) if aggregates["tokens"] else 0.0,
        "mean_bits_per_sample": (aggregates["bits"] / aggregates["samples"]) if aggregates["samples"] else 0.0,
        "image_bits_total": aggregates["image_bits"],
        "text_bits_total": aggregates["text_bits"],
        "lora_param_counts": lora_counts,
        "output_file": str(jsonl_path),
        "description": args.description,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved sample metrics to", jsonl_path)
    print("Saved run metadata to", meta_path)
    if aggregates["samples"]:
        print(
            f"Processed {aggregates['samples']} samples, {aggregates['tokens']} tokens,"
            f" total surprisal {aggregates['bits']:.2f} bits"
        )


if __name__ == "__main__":
    main()
