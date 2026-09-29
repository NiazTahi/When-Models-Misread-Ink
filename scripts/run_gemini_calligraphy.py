import argparse
import base64
import io
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image
from tqdm import tqdm


LABELS = ["Kufic", "Naskh", "Thuluth", "Diwani", "Ruq'ah", "Nasta'liq"]

PROMPTS = {
    "open_ended": "What Arabic calligraphy script is used in this image? Answer with just the script name.",
    "closed_set": "Identify the Arabic calligraphy script. Choose exactly one: Kufic, Naskh, Thuluth, Diwani, Ruq'ah, Nasta'liq.",
    "alt_wording": "Classify the calligraphy style shown in the image. Respond with only one of these labels: Kufic, Naskh, Thuluth, Diwani, Ruq'ah, Nasta'liq.",
    "label_order_rev": "Identify the Arabic calligraphy script. Choose exactly one: Nasta'liq, Ruq'ah, Diwani, Thuluth, Naskh, Kufic.",
}


def split_labels(style):
    return [part.strip() for part in str(style or "").split(",") if part.strip()]


def load_env_file(path):
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("error"):
                completed.add((row.get("model"), row.get("prompt_id"), row.get("image_id")))
    return completed


def image_payload(image_obj, max_image_side):
    data = image_obj["bytes"]
    with Image.open(io.BytesIO(data)) as im:
        mime_type = Image.MIME.get(im.format, "image/jpeg")
        if max_image_side and max(im.size) > max_image_side:
            im.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            fmt = im.format or "JPEG"
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")
            im.save(out, format=fmt)
            data = out.getvalue()
            mime_type = Image.MIME.get(fmt, mime_type)
    return data, mime_type


def call_gemini(api_key, model, prompt, image_bytes, mime_type, temperature, max_output_tokens, timeout):
    model_name = model.removeprefix("models/")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    response = requests.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        body = response.text.replace("\n", " ")[:500]
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {body}")
    data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "\n".join(part.get("text", "") for part in parts if part.get("text"))


def run_one(row, prompt_id, args, api_key):
    labels = split_labels(row["style"])
    started = time.time()
    output_text = ""
    error = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            image_bytes, mime_type = image_payload(row["image"], args.max_image_side)
            output_text = call_gemini(
                api_key=api_key,
                model=args.model,
                prompt=PROMPTS[prompt_id],
                image_bytes=image_bytes,
                mime_type=mime_type,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
            )
            error = ""
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                retry_match = re.search(r"retry in ([0-9.]+)s", error, re.IGNORECASE)
                if retry_match:
                    sleep_s = float(retry_match.group(1)) + 2
                else:
                    rate_limited = "HTTP 429" in error or "Too Many Requests" in error
                    base_sleep = 20 if rate_limited else 2
                    sleep_s = min(base_sleep * attempt, 90)
                time.sleep(sleep_s + random.uniform(0, 2))

    return {
        "eval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "prompt_id": prompt_id,
        "prompt": PROMPTS[prompt_id],
        "image_id": row["image_id"],
        "true_label": labels[0] if len(labels) == 1 else None,
        "raw_style": row["style"],
        "is_multilabel": len(labels) > 1,
        "category": row["category"],
        "source": row["source"],
        "text": row["text"],
        "word_count": row["word_count"],
        "total_words": row["total_words"],
        "output_text": output_text,
        "error": error,
        "latency_s": round(time.time() - started, 3),
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Gemini VLM inference on DuwatBench calligraphy images.")
    parser.add_argument("--parquet", default="train-00000-of-00001.parquet")
    parser.add_argument("--out", default="results/gemini_calligraphy_results.jsonl")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--prompts", default="closed_set", help="Comma-separated prompt ids or 'all'.")
    parser.add_argument("--split", choices=["single", "all"], default="single")
    parser.add_argument("--limit", type=int, default=0, help="Maximum dataset rows to consider after filtering; 0 means all.")
    parser.add_argument("--balanced-per-class", type=int, default=0, help="Use the first N single-label rows from each class.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=6.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--max-image-side", type=int, default=0, help="Optional resize cap; 0 sends original bytes.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--env-file", default=".env", help="Local dotenv file to load before reading GEMINI_API_KEY.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY before running Gemini inference.")

    prompt_ids = list(PROMPTS) if args.prompts == "all" else [p.strip() for p in args.prompts.split(",") if p.strip()]
    unknown = [p for p in prompt_ids if p not in PROMPTS]
    if unknown:
        raise SystemExit(f"Unknown prompt id(s): {unknown}. Available: {sorted(PROMPTS)}")

    parquet_path = Path(args.parquet)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(
        parquet_path,
        columns=["image", "image_id", "style", "category", "source", "text", "word_count", "total_words"],
    )
    rows = table.to_pylist()
    filtered = []
    per_class_counts = {label: 0 for label in LABELS}
    for row in rows:
        labels = split_labels(row["style"])
        if args.split == "single" and len(labels) != 1:
            continue
        if args.balanced_per_class:
            if len(labels) != 1 or labels[0] not in per_class_counts:
                continue
            if per_class_counts[labels[0]] >= args.balanced_per_class:
                continue
            per_class_counts[labels[0]] += 1
        filtered.append(row)
    if args.offset:
        filtered = filtered[args.offset :]
    if args.limit:
        filtered = filtered[: args.limit]

    completed = load_completed(out_path)
    total = len(filtered) * len(prompt_ids)
    print(f"model={args.model} rows={len(filtered)} prompts={prompt_ids} planned_pairs={total}")
    print(f"writing={out_path}")

    tasks = []
    for row in filtered:
        for prompt_id in prompt_ids:
            key = (args.model, prompt_id, row["image_id"])
            if key not in completed:
                tasks.append((row, prompt_id))

    with out_path.open("a", encoding="utf-8") as f:
        if args.workers <= 1:
            iterator = ((None, run_one(row, prompt_id, args, api_key)) for row, prompt_id in tasks)
            for _, record in tqdm(iterator, total=len(tasks), desc="Gemini inference"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                if not record.get("error"):
                    completed.add((record.get("model"), record.get("prompt_id"), record.get("image_id")))
                if args.sleep:
                    time.sleep(args.sleep)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_one, row, prompt_id, args, api_key) for row, prompt_id in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Gemini inference"):
                    record = future.result()
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    if not record.get("error"):
                        completed.add((record.get("model"), record.get("prompt_id"), record.get("image_id")))
                    if args.sleep:
                        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
