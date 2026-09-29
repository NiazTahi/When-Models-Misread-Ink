#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"

LABELS = ["Kufic", "Naskh", "Thuluth", "Diwani", "Ruq'ah", "Nasta'liq"]
NUM_TO_LABEL = {
    "1": "Kufic",
    "2": "Naskh",
    "3": "Thuluth",
    "4": "Diwani",
    "5": "Ruq'ah",
    "6": "Nasta'liq",
}
PROMPT_NAMES = {
    "prompt_a": "Numeric",
    "prompt_b": "Visual-cue",
    "prompt_c": "Diwani-first",
}
OPEN_MODELS = {
    "qwen25vl7b": "Qwen2.5-VL-7B",
    "qwen25vl32b": "Qwen2.5-VL-32B",
    "qwen3vl8b": "Qwen3-VL-8B",
    "qwen3vl32b": "Qwen3-VL-32B",
    "internvl35_8b": "InternVL3.5-8B",
    "internvl35_38b": "InternVL3.5-38B",
    "minicpmv45": "MiniCPM-V 4.5",
    "gemma4_12b": "Gemma 4 12B-IT",
}
OPEN_ORDER = list(OPEN_MODELS.values())
CLOSED_FILES = {
    ("GPT-4o", "Numeric"): "numeric4o.csv",
    ("GPT-4.1", "Numeric"): "numeric4.1.csv",
    ("GPT-5.5", "Numeric"): "numeric5.5.csv",
    ("GPT-5.6 Terra", "Numeric"): "numeric5.6.csv",
    ("GPT-4o", "Visual-cue"): "visual4o.csv",
    ("GPT-4.1", "Visual-cue"): "visual4.1.csv",
    ("GPT-5.5", "Visual-cue"): "visual5.5.csv",
    ("GPT-5.6 Terra", "Visual-cue"): "visual5.6.csv",
    ("GPT-4o", "Diwani-first"): "reordered4o.csv",
    ("GPT-4.1", "Diwani-first"): "reordered4.1.csv",
    ("GPT-5.5", "Diwani-first"): "reordered5.5.csv",
    ("GPT-5.6 Terra", "Diwani-first"): "reordered5.6.csv",
}
CLOSED_ORDER = ["GPT-4o", "GPT-4.1", "GPT-5.5", "GPT-5.6 Terra"]


def norm_text(text: str | None) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return text.replace("’", "'").replace("`", "'").replace("â€™", "'")


ALIASES = {
    "Kufic": [r"\bkufic\b", r"\bkufi\b"],
    "Naskh": [r"\bnaskh\b", r"\bnaskhi\b", r"\bnasakh\b"],
    "Thuluth": [r"\bthuluth\b", r"\bthulth\b", r"\bsulus\b"],
    "Diwani": [r"\bdiwani\b", r"\bdivani\b", r"\bdiwani jali\b", r"\bjali diwani\b"],
    "Ruq'ah": [r"\bruq[' ]?ah\b", r"\bruqaa\b", r"\breqa\b", r"\briq[ae]?\b"],
    "Nasta'liq": [r"\bnasta[' ]?liq\b", r"\bnastaliq\b", r"\bnastaleeq\b", r"\bnastaaliq\b"],
}


def normalize_label(value: str | None) -> str:
    raw = (value or "").strip()
    if raw in LABELS:
        return raw
    if raw in NUM_TO_LABEL:
        return NUM_TO_LABEL[raw]
    text = norm_text(raw)
    hits = [label for label, pats in ALIASES.items() if any(re.search(p, text) for p in pats)]
    return hits[0] if len(hits) == 1 else "Unparsed"


def read_open_rows() -> list[dict]:
    rows = []
    for key, model_name in OPEN_MODELS.items():
        model_dir = ROOT / "open_source" / key
        for prompt_id, prompt_name in PROMPT_NAMES.items():
            prompt_letter = prompt_id[-1]
            path = model_dir / f"prompt_{prompt_letter}" / f"{key}_prompt_{prompt_letter}.jsonl"
            if not path.exists():
                rows.append({"model_group": "Open-weight", "model": model_name, "prompt": prompt_name, "missing": str(path)})
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    rows.append(
                        {
                            "model_group": "Open-weight",
                            "model": model_name,
                            "prompt": prompt_name,
                            "filename": item.get("filename", ""),
                            "true_label": normalize_label(item.get("true_label") or item.get("style")),
                            "pred_label": normalize_label(item.get("output_text")),
                            "raw_answer": item.get("output_text", ""),
                            "source_dataset": item.get("source_dataset", ""),
                            "category": item.get("category", ""),
                            "error": item.get("error", ""),
                        }
                    )
    return rows


def read_closed_rows() -> list[dict]:
    rows = []
    base = ROOT / "colleague_exp_result" / "merged_results" / "merged_results"
    for (model, prompt), filename in CLOSED_FILES.items():
        path = base / filename
        if not path.exists():
            rows.append({"model_group": "Closed/API", "model": model, "prompt": prompt, "missing": str(path)})
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for item in csv.DictReader(f):
                rows.append(
                    {
                        "model_group": "Closed/API",
                        "model": model,
                        "prompt": prompt,
                        "filename": item.get("filename", ""),
                        "true_label": normalize_label(item.get("true_label")),
                        "pred_label": normalize_label(item.get("ai_answer_mapped") or item.get("ai_answer_raw")),
                        "raw_answer": item.get("ai_answer_raw", ""),
                        "source_dataset": item.get("source_dataset", ""),
                        "category": "",
                        "error": "",
                    }
                )
    return rows


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def summarize(rows: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    n = len(rows)
    correct = sum(r["true_label"] == r["pred_label"] for r in rows)
    true_counts = Counter(r["true_label"] for r in rows)
    pred_counts = Counter(r["pred_label"] for r in rows)
    per_class = []
    recalls = []
    f1s = []
    for label in LABELS:
        tp = sum(r["true_label"] == label and r["pred_label"] == label for r in rows)
        fp = sum(r["true_label"] != label and r["pred_label"] == label for r in rows)
        fn = sum(r["true_label"] == label and r["pred_label"] != label for r in rows)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        recalls.append(recall)
        f1s.append(f1)
        per_class.append(
            {
                "label": label,
                "support": true_counts[label],
                "predicted": pred_counts[label],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "prediction_ratio": safe_div(pred_counts[label], true_counts[label]),
            }
        )
    summary = {
        "n": n,
        "accuracy": safe_div(correct, n),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "parse_error_rate": safe_div(sum(r["pred_label"] == "Unparsed" for r in rows), n),
        "thuluth_ratio": safe_div(pred_counts["Thuluth"], true_counts["Thuluth"]),
        "diwani_ratio": safe_div(pred_counts["Diwani"], true_counts["Diwani"]),
        "diwani_recall": next(x["recall"] for x in per_class if x["label"] == "Diwani"),
        "ruqah_recall": next(x["recall"] for x in per_class if x["label"] == "Ruq'ah"),
    }
    confusion = []
    for true in LABELS:
        row = {"true_label": true}
        for pred in LABELS + ["Unparsed"]:
            row[pred] = sum(r["true_label"] == true and r["pred_label"] == pred for r in rows)
        confusion.append(row)
    return summary, per_class, confusion


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(x: float) -> str:
    return f"{100*x:.1f}"


def main() -> None:
    all_rows = [r for r in read_closed_rows() + read_open_rows() if not r.get("missing")]
    missing = [r for r in read_closed_rows() + read_open_rows() if r.get("missing")]
    summary_rows = []
    per_class_rows = []
    confusion_rows = []
    for key in sorted({(r["model_group"], r["model"], r["prompt"]) for r in all_rows}):
        group_rows = [r for r in all_rows if (r["model_group"], r["model"], r["prompt"]) == key]
        summary, per_class, confusion = summarize(group_rows)
        model_group, model, prompt = key
        summary_rows.append({"model_group": model_group, "model": model, "prompt": prompt, **summary})
        for row in per_class:
            per_class_rows.append({"model_group": model_group, "model": model, "prompt": prompt, **row})
        for row in confusion:
            confusion_rows.append({"model_group": model_group, "model": model, "prompt": prompt, **row})

    write_csv(
        OUT / "unified_summary_metrics.csv",
        summary_rows,
        [
            "model_group",
            "model",
            "prompt",
            "n",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "parse_error_rate",
            "thuluth_ratio",
            "diwani_ratio",
            "diwani_recall",
            "ruqah_recall",
        ],
    )
    write_csv(
        OUT / "unified_per_class_metrics.csv",
        per_class_rows,
        ["model_group", "model", "prompt", "label", "support", "predicted", "precision", "recall", "f1", "prediction_ratio"],
    )
    write_csv(
        OUT / "unified_confusion_counts.csv",
        confusion_rows,
        ["model_group", "model", "prompt", "true_label", *LABELS, "Unparsed"],
    )
    write_csv(OUT / "unified_all_predictions.csv", all_rows, ["model_group", "model", "prompt", "filename", "true_label", "pred_label", "raw_answer", "source_dataset", "category", "error"])
    if missing:
        write_csv(OUT / "missing_expected_files.csv", missing, ["model_group", "model", "prompt", "missing"])

    order_model = {m: i for i, m in enumerate(CLOSED_ORDER + OPEN_ORDER)}
    order_prompt = {"Numeric": 0, "Visual-cue": 1, "Diwani-first": 2}
    summary_rows.sort(key=lambda r: (r["model_group"] != "Closed/API", order_model.get(r["model"], 999), order_prompt.get(r["prompt"], 999)))
    md = []
    md.append("# Unified Calligraphy Analysis\n")
    md.append("Generated from colleague closed-model CSVs and local open-weight JSONLs.\n")
    md.append("\n## Dataset and result coverage\n")
    md.append(f"- Images per run: {summary_rows[0]['n'] if summary_rows else 0}\n")
    md.append(f"- Completed model-prompt runs: {len(summary_rows)}\n")
    md.append(f"- Missing expected files: {len(missing)}\n")
    md.append("\n## Main summary table\n")
    md.append("| Group | Model | Prompt | Accuracy (%) | Balanced acc. (%) | Macro-F1 | Thuluth prediction ratio | Diwani recall (%) | Parse errors (%) |\n")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in summary_rows:
        md.append(
            f"| {r['model_group']} | {r['model']} | {r['prompt']} | {fmt_pct(r['accuracy'])} | {fmt_pct(r['balanced_accuracy'])} | {r['macro_f1']:.3f} | {r['thuluth_ratio']:.2f}x | {fmt_pct(r['diwani_recall'])} | {fmt_pct(r['parse_error_rate'])} |\n"
        )
    md.append("\n## Strongest paper-relevant facts\n")
    numeric = [r for r in summary_rows if r["prompt"] == "Numeric"]
    visual = [r for r in summary_rows if r["prompt"] == "Visual-cue"]
    diwani_first = [r for r in summary_rows if r["prompt"] == "Diwani-first"]
    if numeric:
        best = max(numeric, key=lambda r: r["accuracy"])
        worst_diwani = min(numeric, key=lambda r: r["diwani_recall"])
        th_over = sum(1 for r in numeric if r["thuluth_ratio"] > 1.0)
        md.append(f"- Under the numeric prompt, best accuracy is {fmt_pct(best['accuracy'])}% ({best['model']}); lowest Diwani recall is {fmt_pct(worst_diwani['diwani_recall'])}% ({worst_diwani['model']}).\n")
        md.append(f"- Thuluth is over-predicted in {th_over}/{len(numeric)} numeric-prompt runs: min ratio {min(r['thuluth_ratio'] for r in numeric):.2f}x, max ratio {max(r['thuluth_ratio'] for r in numeric):.2f}x.\n")
    if visual:
        deltas = []
        for r in visual:
            base = next((b for b in numeric if b["model"] == r["model"] and b["model_group"] == r["model_group"]), None)
            if base:
                deltas.append((r["accuracy"] - base["accuracy"], r, base))
        if deltas:
            up = max(deltas, key=lambda x: x[0])
            down = min(deltas, key=lambda x: x[0])
            md.append(f"- Visual cues have nonuniform effects: largest gain {100*up[0]:+.1f} points ({up[1]['model']}), largest drop {100*down[0]:+.1f} points ({down[1]['model']}).\n")
    if diwani_first:
        deltas = []
        for r in diwani_first:
            base = next((b for b in visual if b["model"] == r["model"] and b["model_group"] == r["model_group"]), None)
            if base:
                deltas.append((r["diwani_recall"] - base["diwani_recall"], r, base))
        if deltas:
            best_shift = max(deltas, key=lambda x: x[0])
            md.append(f"- Moving Diwani first produces the largest Diwani-recall gain for {best_shift[1]['model']}: {100*best_shift[0]:+.1f} points.\n")
    (OUT / "UNIFIED_ANALYSIS.md").write_text("".join(md), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"runs={len(summary_rows)} missing={len(missing)}")


if __name__ == "__main__":
    main()
