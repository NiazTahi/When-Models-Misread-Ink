import argparse
import base64
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from run_gemini_calligraphy import LABELS, PROMPTS, load_env_file, split_labels


TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def safe_key(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def image_payload_for_batch(image_obj, max_image_side, jpeg_quality):
    data = image_obj["bytes"]
    with Image.open(io.BytesIO(data)) as im:
        if max_image_side and max(im.size) > max_image_side:
            im.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=jpeg_quality, optimize=True)
        return out.getvalue(), "image/jpeg"


def iter_filtered_rows(parquet_path, split, balanced_per_class, limit, offset):
    table = pq.read_table(
        parquet_path,
        columns=["image", "image_id", "style", "category", "source", "text", "word_count", "total_words"],
    )
    rows = table.to_pylist()
    filtered = []
    per_class_counts = {label: 0 for label in LABELS}
    for row in rows:
        labels = split_labels(row["style"])
        if split == "single" and len(labels) != 1:
            continue
        if balanced_per_class:
            if len(labels) != 1 or labels[0] not in per_class_counts:
                continue
            if per_class_counts[labels[0]] >= balanced_per_class:
                continue
            per_class_counts[labels[0]] += 1
        filtered.append(row)
    if offset:
        filtered = filtered[offset:]
    if limit:
        filtered = filtered[:limit]
    return filtered


def make_jsonl(args):
    prompt_ids = list(PROMPTS) if args.prompts == "all" else [p.strip() for p in args.prompts.split(",") if p.strip()]
    unknown = [p for p in prompt_ids if p not in PROMPTS]
    if unknown:
        raise SystemExit(f"Unknown prompt id(s): {unknown}. Available: {sorted(PROMPTS)}")

    rows = iter_filtered_rows(
        parquet_path=Path(args.parquet),
        split=args.split,
        balanced_per_class=args.balanced_per_class,
        limit=args.limit,
        offset=args.offset,
    )
    out_path = Path(args.out)
    sidecar_path = Path(args.sidecar or f"{out_path}.metadata.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as out_f, sidecar_path.open("w", encoding="utf-8") as meta_f:
        for row in tqdm(rows, desc="Writing batch JSONL"):
            labels = split_labels(row["style"])
            for prompt_id in prompt_ids:
                key = safe_key(f"{args.model}__{prompt_id}__{row['image_id']}")
                image_bytes, mime_type = image_payload_for_batch(row["image"], args.max_image_side, args.jpeg_quality)
                request = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": PROMPTS[prompt_id]},
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": base64.b64encode(image_bytes).decode("ascii"),
                                    }
                                },
                            ],
                        }
                    ],
                    "generation_config": {
                        "temperature": args.temperature,
                        "max_output_tokens": args.max_output_tokens,
                    },
                }
                out_f.write(json.dumps({"key": key, "request": request}, ensure_ascii=False) + "\n")
                meta_f.write(
                    json.dumps(
                        {
                            "key": key,
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
                            "temperature": args.temperature,
                            "max_output_tokens": args.max_output_tokens,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1

    print(f"requests={n}")
    print(f"batch_jsonl={out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")
    print(f"metadata={sidecar_path}")


def client_from_env(env_file):
    load_env_file(env_file)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env first.")
    from google import genai

    return genai.Client(api_key=api_key)


def submit(args):
    from google.genai import types

    client = client_from_env(args.env_file)
    uploaded_file = client.files.upload(
        file=args.input_jsonl,
        config=types.UploadFileConfig(display_name=args.display_name, mime_type="jsonl"),
    )
    print(f"uploaded_file_name={uploaded_file.name}")
    print(f"uploaded_file_uri={getattr(uploaded_file, 'uri', '')}")
    job = client.batches.create(
        model=args.model,
        src=uploaded_file.name,
        config={"display_name": args.display_name},
    )
    print(f"batch_job_name={job.name}")
    print(f"state={getattr(job.state, 'name', job.state)}")
    job_path = Path(args.job_file)
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "input_jsonl": str(args.input_jsonl),
                "uploaded_file_name": uploaded_file.name,
                "batch_job_name": job.name,
                "display_name": args.display_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"job_file={job_path}")


def status(args):
    client = client_from_env(args.env_file)
    job_name = args.job_name
    if args.job_file and not job_name:
        job_name = json.loads(Path(args.job_file).read_text(encoding="utf-8"))["batch_job_name"]
    if not job_name:
        raise SystemExit("Pass --job-name or --job-file.")
    while True:
        job = client.batches.get(name=job_name)
        state = getattr(job.state, "name", str(job.state))
        print(f"{datetime.now().isoformat(timespec='seconds')} {job.name} {state}")
        print(f"stats={getattr(job, 'batch_stats', None) or getattr(job, 'batchStats', None)}")
        print(f"dest={getattr(job, 'dest', None)}")
        if not args.watch or state in TERMINAL_STATES:
            break
        time.sleep(args.poll_seconds)


def download(args):
    client = client_from_env(args.env_file)
    job_name = args.job_name
    if args.job_file and not job_name:
        job_name = json.loads(Path(args.job_file).read_text(encoding="utf-8"))["batch_job_name"]
    if not job_name:
        raise SystemExit("Pass --job-name or --job-file.")
    job = client.batches.get(name=job_name)
    state = getattr(job.state, "name", str(job.state))
    if state != "JOB_STATE_SUCCEEDED":
        raise SystemExit(f"Batch is not succeeded yet: {state}")
    result_file_name = job.dest.file_name
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client.files.download(file=result_file_name, destination=out_path)
    print(f"downloaded={out_path}")


def extract_text(response_obj):
    parts = response_obj.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "\n".join(part.get("text", "") for part in parts if part.get("text"))


def convert(args):
    metadata = {}
    with Path(args.sidecar).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                metadata[row["key"]] = row

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    errors = 0
    with Path(args.batch_output).open("r", encoding="utf-8") as in_f, out_path.open("w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key") or row.get("metadata", {}).get("key")
            base = metadata.get(key, {"key": key})
            response = row.get("response") or row.get("inlineResponse") or {}
            error = row.get("error") or response.get("error") or ""
            if error:
                errors += 1
            record = {
                **base,
                "eval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "output_text": extract_text(response),
                "error": json.dumps(error, ensure_ascii=False) if isinstance(error, dict) else str(error),
                "latency_s": None,
                "batch_key": key,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"converted={n}")
    print(f"errors={errors}")
    print(f"results_jsonl={out_path}")


def main():
    parser = argparse.ArgumentParser(description="Gemini Batch API workflow for calligraphy inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_p = subparsers.add_parser("make-jsonl")
    make_p.add_argument("--parquet", default="train-00000-of-00001.parquet")
    make_p.add_argument("--out", required=True)
    make_p.add_argument("--sidecar", default=None)
    make_p.add_argument("--model", default="gemini-3.5-flash-lite")
    make_p.add_argument("--prompts", default="closed_set")
    make_p.add_argument("--split", choices=["single", "all"], default="single")
    make_p.add_argument("--limit", type=int, default=0)
    make_p.add_argument("--offset", type=int, default=0)
    make_p.add_argument("--balanced-per-class", type=int, default=0)
    make_p.add_argument("--temperature", type=float, default=0.0)
    make_p.add_argument("--max-output-tokens", type=int, default=64)
    make_p.add_argument("--max-image-side", type=int, default=1536)
    make_p.add_argument("--jpeg-quality", type=int, default=90)
    make_p.set_defaults(func=make_jsonl)

    submit_p = subparsers.add_parser("submit")
    submit_p.add_argument("--input-jsonl", required=True)
    submit_p.add_argument("--model", default="gemini-3.5-flash-lite")
    submit_p.add_argument("--display-name", default="duwatbench-calligraphy")
    submit_p.add_argument("--job-file", default="results/gemini_batch_job.json")
    submit_p.add_argument("--env-file", default=".env")
    submit_p.set_defaults(func=submit)

    status_p = subparsers.add_parser("status")
    status_p.add_argument("--job-name", default=None)
    status_p.add_argument("--job-file", default=None)
    status_p.add_argument("--env-file", default=".env")
    status_p.add_argument("--watch", action="store_true")
    status_p.add_argument("--poll-seconds", type=int, default=60)
    status_p.set_defaults(func=status)

    download_p = subparsers.add_parser("download")
    download_p.add_argument("--job-name", default=None)
    download_p.add_argument("--job-file", default=None)
    download_p.add_argument("--out", required=True)
    download_p.add_argument("--env-file", default=".env")
    download_p.set_defaults(func=download)

    convert_p = subparsers.add_parser("convert")
    convert_p.add_argument("--batch-output", required=True)
    convert_p.add_argument("--sidecar", required=True)
    convert_p.add_argument("--out", required=True)
    convert_p.set_defaults(func=convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
