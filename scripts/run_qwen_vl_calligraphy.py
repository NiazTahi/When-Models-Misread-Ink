#!/usr/bin/env python3
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def read_completed(output_path: Path) -> set[str]:
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            filename = row.get("filename")
            if filename:
                completed.add(filename)
    return completed


def load_rows(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_messages(prompt: str, image_path: Path) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def infer_one(model, processor, prompt: str, image_path: Path, max_new_tokens: int) -> str:
    # Verify image readability before passing it into the processor.
    with Image.open(image_path) as img:
        img.verify()

    messages = build_messages(prompt, image_path)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_reproducibility(args.seed)

    manifest_path = args.dataset_root / "merged_manifest.csv"
    image_dir = args.dataset_root / "images"
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(args.output)
    rows = load_rows(manifest_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"model={args.model_id}")
    print(f"seed={args.seed} do_sample=False num_beams=1")
    print(f"rows={len(rows)} completed={len(completed)} output={args.output}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    with args.output.open("a", encoding="utf-8") as out:
        for row in tqdm(rows):
            filename = row["filename"]
            if filename in completed:
                continue
            image_path = image_dir / filename
            started = time.time()
            result = {
                "model": args.model_id,
                "prompt_id": args.prompt_id,
                "filename": filename,
                "image_id": filename,
                "true_label": row.get("style", ""),
                "style": row.get("style", ""),
                "source_dataset": row.get("source_dataset", ""),
                "category": row.get("category", ""),
                "text": row.get("text", ""),
                "output_text": "",
                "error": "",
                "latency_sec": None,
            }
            try:
                result["output_text"] = infer_one(
                    model, processor, prompt, image_path, args.max_new_tokens
                )
            except Exception as exc:
                result["error"] = repr(exc)
            result["latency_sec"] = round(time.time() - started, 3)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
