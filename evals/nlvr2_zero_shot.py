"""Zero-shot evaluation of Qwen2-VL-2B on NLVR2."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.nlvr2 import (
    NLVR2DataConfig,
    build_chat_messages,
    load_nlvr2,
    materialize_dataset,
)

logger = logging.getLogger(__name__)


def _prepare_generation_batch(processor, batch: List[dict]) -> dict:
    input_ids, attention_masks, pixel_values = [], [], []
    for example in batch:
        messages = build_chat_messages(example["sentence"], example["images"])
        processed = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids.append(processed["input_ids"][0])
        attention_masks.append(processed["attention_mask"][0])
        pixel_values.append(processed["pixel_values"][0])

    padded = processor.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,
        return_tensors="pt",
    )
    pixels = torch.stack(pixel_values, dim=0)
    padded["pixel_values"] = pixels
    return padded


def _extract_prediction(text: str) -> str:
    text = text.strip()
    lowered = text.lower()
    if lowered.endswith("true"):
        return "True"
    if lowered.endswith("false"):
        return "False"
    if "true" in lowered and lowered.rfind("true") > lowered.rfind("false"):
        return "True"
    if "false" in lowered and lowered.rfind("false") > lowered.rfind("true"):
        return "False"
    return "Unknown"


def evaluate(
    model_name: str,
    data_config: NLVR2DataConfig,
    limit: Optional[int],
    batch_size: int,
    output_path: Optional[str] = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    dataset = load_nlvr2(data_config)
    examples_dataset = materialize_dataset(dataset, limit)
    examples = [examples_dataset[i] for i in range(len(examples_dataset))]

    correct = 0
    total = 0
    records = []

    for idx in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
        batch = examples[idx : idx + batch_size]
        batch_inputs = _prepare_generation_batch(processor, batch)
        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

        with torch.no_grad():
            output_tokens = model.generate(
                **batch_inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        input_lengths = batch_inputs["attention_mask"].sum(dim=1)
        for i, example in enumerate(batch):
            generated_tokens = output_tokens[i, input_lengths[i] :]
            text = processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            prediction = _extract_prediction(text)
            gold = example["label"]
            is_correct = prediction == gold
            correct += int(is_correct)
            total += 1
            record = {
                "uid": example.get("uid"),
                "sentence": example.get("sentence"),
                "prediction": prediction,
                "label": gold,
                "raw_text": text,
                "correct": is_correct,
            }
            if example.get("metadata"):
                record["metadata"] = example["metadata"]
            records.append(record)

    accuracy = correct / max(total, 1)
    logger.info("Accuracy: %.4f (%d/%d)", accuracy, correct, total)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        logger.info("Wrote predictions to %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-shot NLVR2 evaluation")
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=512, help="Number of examples to evaluate")
    parser.add_argument("--streaming", action="store_true", help="Stream data instead of downloading")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=str, default="logs/zero_shot_predictions.jsonl")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle streaming dataset")
    parser.add_argument("--streaming-take", type=int, default=None, help="Cap number of streaming examples before materializing")
    parser.add_argument("--randomize-image-order", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
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
