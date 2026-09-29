#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel, AutoTokenizer


if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    def _get_all_tied_weights_keys(self):
        value = getattr(self, "_all_tied_weights_keys_compat", None)
        if value is not None:
            return value
        return {key: key for key in (getattr(self, "_tied_weights_keys", None) or [])}

    def _set_all_tied_weights_keys(self, value):
        self._all_tied_weights_keys_compat = value

    PreTrainedModel.all_tied_weights_keys = property(
        _get_all_tied_weights_keys, _set_all_tied_weights_keys
    )


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


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_model(model_id: str):
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tokenizer, model


def call_chat(model, tokenizer, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": prompt}]
    requested = {
        "image": image,
        "msgs": msgs,
        "tokenizer": tokenizer,
        "enable_thinking": False,
        "stream": False,
        "sampling": False,
        "max_new_tokens": max_new_tokens,
    }
    signature = inspect.signature(model.chat)
    has_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    if has_var_kwargs:
        kwargs = requested
    else:
        kwargs = {key: value for key, value in requested.items() if key in signature.parameters}
    answer = model.chat(**kwargs)
    if isinstance(answer, tuple):
        answer = answer[0]
    if not isinstance(answer, str):
        answer = "".join(answer)
    return answer.strip()


def infer_one(model, tokenizer, prompt: str, image_path: Path, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    try:
        return call_chat(model, tokenizer, image, prompt, max_new_tokens)
    finally:
        image.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--model-id", required=True)
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

    print(f"model={args.model_id}", flush=True)
    print(f"seed={args.seed} enable_thinking=False sampling=False num_beams=1", flush=True)
    print(f"rows={len(rows)} completed={len(completed)} output={args.output}", flush=True)

    tokenizer, model = load_model(args.model_id)

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
                    model, tokenizer, prompt, image_path, args.max_new_tokens
                )
            except Exception as exc:
                result["error"] = repr(exc)
            result["latency_sec"] = round(time.time() - started, 3)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
