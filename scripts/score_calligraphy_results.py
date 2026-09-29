import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = ["Kufic", "Naskh", "Thuluth", "Diwani", "Ruq'ah", "Nasta'liq"]
OTHER = "Unparsed"

ALIASES = {
    "Kufic": [r"\bkufic\b", r"\bkufi\b", r"\bmodern kufic\b"],
    "Naskh": [r"\bnaskh\b", r"\bnaskhi\b", r"\bnasakh\b"],
    "Thuluth": [r"\bthuluth\b", r"\bthulth\b", r"\bsulus\b"],
    "Diwani": [r"\bdiwani\b", r"\bdivani\b", r"\bdiwani jali\b", r"\bjali diwani\b"],
    "Ruq'ah": [r"\bruq['’` ]?ah\b", r"\bruqaa\b", r"\breqa\b", r"\briq[ae]?\b"],
    "Nasta'liq": [r"\bnasta['’` ]?liq\b", r"\bnastaliq\b", r"\bnastaleeq\b", r"\bnastaaliq\b"],
}

REFUSAL_RE = re.compile(r"\b(can't|cannot|unable|sorry|not able|refuse|inappropriate|policy)\b", re.I)
HEDGE_RE = re.compile(r"\b(maybe|perhaps|appears|seems|likely|probably|not sure|uncertain|hard to tell)\b", re.I)


def canonicalize_text(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = text.replace("’", "'").replace("`", "'")
    return text


def normalize_label(text):
    norm = canonicalize_text(text)
    hits = []
    for label, patterns in ALIASES.items():
        if any(re.search(pattern, norm) for pattern in patterns):
            hits.append(label)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "Multiple"
    cleaned = re.sub(r"[^a-z' ]+", " ", norm).strip()
    for label in LABELS:
        if cleaned == canonicalize_text(label):
            return label
    return OTHER


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def safe_div(num, den):
    return num / den if den else 0.0


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def summarize_group(rows):
    n = len(rows)
    y_true = [r["true_label"] for r in rows]
    y_pred = [r["pred_label"] for r in rows]
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    per_class = []
    recalls = []
    f1s = []
    for label in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
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
            }
        )

    diwani_pred = pred_counts["Diwani"]
    diwani_true = true_counts["Diwani"]
    non_diwani = n - diwani_true
    non_diwani_as_diwani = sum(1 for t, p in zip(y_true, y_pred) if t != "Diwani" and p == "Diwani")
    return {
        "n": n,
        "accuracy": safe_div(correct, n),
        "majority_baseline": safe_div(max(true_counts.values()) if true_counts else 0, n),
        "balanced_accuracy": sum(recalls) / len(LABELS),
        "macro_f1": sum(f1s) / len(LABELS),
        "parse_error_rate": safe_div(sum(1 for p in y_pred if p in {OTHER, "Multiple"}), n),
        "diwani_true_rate": safe_div(diwani_true, n),
        "diwani_pred_rate": safe_div(diwani_pred, n),
        "diwani_overprediction_ratio": safe_div(safe_div(diwani_pred, n), safe_div(diwani_true, n)),
        "non_diwani_as_diwani_rate": safe_div(non_diwani_as_diwani, non_diwani),
        "per_class": per_class,
        "true_counts": true_counts,
        "pred_counts": pred_counts,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path, rows):
    columns = LABELS + ["Multiple", OTHER]
    matrix = []
    for true_label in LABELS:
        line = {"true_label": true_label}
        for pred_label in columns:
            line[pred_label] = sum(1 for r in rows if r["true_label"] == true_label and r["pred_label"] == pred_label)
        matrix.append(line)
    write_csv(path, matrix, ["true_label"] + columns)


def plot_confusion(path, rows, title):
    columns = LABELS + ["Multiple", OTHER]
    values = [[sum(1 for r in rows if r["true_label"] == t and r["pred_label"] == p) for p in columns] for t in LABELS]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i, row in enumerate(values):
        for j, val in enumerate(row):
            ax.text(j, i, str(val), ha="center", va="center", color="black" if val < max(max(r) for r in values) / 2 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def category_flags(row):
    category = canonicalize_text(row.get("category", ""))
    religious = category != "non-religious"
    output = row.get("output_text", "")
    return {
        "religious": religious,
        "refusal": bool(REFUSAL_RE.search(output)),
        "hedged": bool(HEDGE_RE.search(output)),
        "empty": not output.strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Score Gemini/OpenAI calligraphy style results.")
    parser.add_argument("--results", default="results/gemini_calligraphy_results.jsonl")
    parser.add_argument("--out-dir", default="results/scored_gemini")
    parser.add_argument("--include-errors", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows = []
    for row in read_jsonl(args.results):
        if row.get("is_multilabel"):
            continue
        if row.get("error") and not args.include_errors:
            continue
        true_label = normalize_label(row.get("true_label") or row.get("raw_style"))
        if true_label not in LABELS:
            continue
        output = row.get("output_text", "")
        rows.append({**row, "true_label": true_label, "pred_label": normalize_label(output)})

    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("model", ""), row.get("prompt_id", ""))].append(row)

    summary_rows = []
    per_class_rows = []
    content_rows = []
    for (model, prompt_id), items in sorted(groups.items()):
        summary = summarize_group(items)
        summary_rows.append(
            {
                "model": model,
                "prompt_id": prompt_id,
                **{k: summary[k] for k in [
                    "n",
                    "accuracy",
                    "majority_baseline",
                    "balanced_accuracy",
                    "macro_f1",
                    "parse_error_rate",
                    "diwani_true_rate",
                    "diwani_pred_rate",
                    "diwani_overprediction_ratio",
                    "non_diwani_as_diwani_rate",
                ]},
            }
        )
        for item in summary["per_class"]:
            per_class_rows.append({"model": model, "prompt_id": prompt_id, **item})

        stem = f"{model}_{prompt_id}".replace("/", "_").replace(":", "_")
        write_confusion(out_dir / f"confusion_{stem}.csv", items)
        plot_confusion(out_dir / f"confusion_{stem}.png", items, f"{model} / {prompt_id}")

        for religious in [True, False]:
            subset = [r for r in items if category_flags(r)["religious"] == religious]
            for metric in ["refusal", "hedged", "empty"]:
                k = sum(1 for r in subset if category_flags(r)[metric])
                lo, hi = wilson_ci(k, len(subset))
                content_rows.append(
                    {
                        "model": model,
                        "prompt_id": prompt_id,
                        "religious": religious,
                        "metric": metric,
                        "n": len(subset),
                        "count": k,
                        "rate": safe_div(k, len(subset)),
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )

    write_csv(
        out_dir / "summary_metrics.csv",
        summary_rows,
        [
            "model",
            "prompt_id",
            "n",
            "accuracy",
            "majority_baseline",
            "balanced_accuracy",
            "macro_f1",
            "parse_error_rate",
            "diwani_true_rate",
            "diwani_pred_rate",
            "diwani_overprediction_ratio",
            "non_diwani_as_diwani_rate",
        ],
    )
    write_csv(out_dir / "per_class_metrics.csv", per_class_rows, ["model", "prompt_id", "label", "support", "predicted", "precision", "recall", "f1"])
    write_csv(out_dir / "religious_content_flags.csv", content_rows, ["model", "prompt_id", "religious", "metric", "n", "count", "rate", "ci95_low", "ci95_high"])

    print(f"scored_rows={len(rows)} groups={len(groups)}")
    print(f"summary={out_dir / 'summary_metrics.csv'}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
