import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from score_calligraphy_results import (
    LABELS,
    OTHER,
    normalize_label,
    read_jsonl,
    summarize_group,
    write_csv,
    write_confusion,
    plot_confusion,
)


def filename_part(value):
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(".", "")
        .replace("-", "")
    )


def prompt_short_name(prompt_id):
    return {
        "open_ended": "pilot_results",
        "closed_set": "phase2_prompt_b",
        "alt_wording": "phase2_prompt_c",
        "label_order_rev": "phase2_prompt_d",
    }.get(prompt_id, f"results_{filename_part(prompt_id)}")


def true_norms(row):
    raw = row.get("true_label") or row.get("raw_style") or ""
    labels = [normalize_label(part) for part in str(raw).split(",")]
    return [label for label in labels if label in LABELS]


def row_outcome(norm_true, pred_label):
    return "correct" if pred_label in norm_true else "incorrect"


def plot_prediction_distribution(path, rows, title):
    labels = LABELS + ["Multiple", OTHER]
    n = len(rows)
    true_counts = {label: 0 for label in labels}
    pred_counts = {label: 0 for label in labels}

    for row in rows:
        norm_true = row["true_norm_list"]
        if len(norm_true) == 1:
            true_counts[norm_true[0]] += 1
        elif len(norm_true) > 1:
            true_counts["Multiple"] += 1
        else:
            true_counts[OTHER] += 1
        pred_counts[row["pred_label"]] = pred_counts.get(row["pred_label"], 0) + 1

    true_pct = [(true_counts[label] / n * 100) if n else 0 for label in labels]
    pred_pct = [(pred_counts[label] / n * 100) if n else 0 for label in labels]

    x = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - width / 2 for i in x], true_pct, width, label="True distribution")
    ax.bar([i + width / 2 for i in x], pred_pct, width, label="Prediction distribution")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("% of rows")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_detail_row(index, row):
    norm_true = true_norms(row)
    pred_label = normalize_label(row.get("output_text", ""))
    return {
        "index": index,
        "image_id": row.get("image_id", ""),
        "model": row.get("model", ""),
        "resolved_model": row.get("resolved_model", ""),
        "prompt_id": row.get("prompt_id", ""),
        "true_label": row.get("true_label") or row.get("raw_style", ""),
        "ai_answer": row.get("output_text", ""),
        "outcome": row_outcome(norm_true, pred_label),
        "true_norm": json.dumps(norm_true, ensure_ascii=False),
        "ai_norm": pred_label,
        "is_multilabel": len(norm_true) > 1 or bool(row.get("is_multilabel")),
        "category": row.get("category", ""),
        "source": row.get("source", ""),
        "total_words": row.get("total_words", ""),
        "latency_s": row.get("latency_s", ""),
        "response_id": row.get("response_id", ""),
        "error": row.get("error", ""),
        "_raw": row,
        "true_norm_list": norm_true,
        "pred_label": pred_label,
    }


def public_row(row, include_outcome=True):
    result = {
        "index": row["index"],
        "true_label": row["true_label"],
        "ai_answer": row["ai_answer"],
    }
    if include_outcome:
        result["outcome"] = row["outcome"]
    return result


def csv_ready(row):
    return {k: v for k, v in row.items() if not k.startswith("_") and k not in {"true_norm_list", "pred_label"}}


def main():
    parser = argparse.ArgumentParser(description="Export EVAN-style detailed calligraphy outputs.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--include-errors", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detailed_rows = []
    for index, row in enumerate(read_jsonl(args.results)):
        if row.get("error") and not args.include_errors:
            continue
        detail = build_detail_row(index, row)
        if detail["true_norm_list"]:
            detailed_rows.append(detail)

    groups = defaultdict(list)
    for row in detailed_rows:
        groups[(row["model"], row["prompt_id"])].append(row)

    all_fieldnames = [
        "index",
        "image_id",
        "model",
        "resolved_model",
        "prompt_id",
        "true_label",
        "ai_answer",
        "outcome",
        "true_norm",
        "ai_norm",
        "is_multilabel",
        "category",
        "source",
        "total_words",
        "latency_s",
        "response_id",
        "error",
    ]
    write_csv(out_dir / "detailed_rows_all_prompts.csv", [csv_ready(r) for r in detailed_rows], all_fieldnames)

    manifest = []
    for (model, prompt_id), rows in sorted(groups.items()):
        model_part = filename_part(model)
        short = prompt_short_name(prompt_id)
        single_rows = [r for r in rows if not r["is_multilabel"]]
        metric_rows = [
            {**r["_raw"], "true_label": r["true_norm_list"][0], "pred_label": r["pred_label"]}
            for r in single_rows
            if len(r["true_norm_list"]) == 1
        ]

        compact_name = f"{short}_{model_part}.csv"
        full_name = f"{short}_{model_part}_full.csv"
        multilabel_name = f"multilabel_rows_{model_part}_{filename_part(prompt_id)}.csv"
        table_name = f"table2_per_class_metrics_{model_part}_{filename_part(prompt_id)}.csv"
        confusion_name = f"figure1_confusion_matrix_{model_part}_{filename_part(prompt_id)}.png"
        distribution_name = f"figure2_prediction_distribution_{model_part}_{filename_part(prompt_id)}.png"

        write_csv(out_dir / compact_name, [public_row(r, include_outcome=False) for r in rows], ["index", "true_label", "ai_answer"])
        write_csv(out_dir / full_name, [csv_ready(r) for r in rows], all_fieldnames)

        multilabel_rows = [r for r in rows if r["is_multilabel"]]
        write_csv(
            out_dir / multilabel_name,
            [csv_ready(r) for r in multilabel_rows],
            all_fieldnames,
        )

        if metric_rows:
            summary = summarize_group(metric_rows)
            write_csv(
                out_dir / table_name,
                summary["per_class"],
                ["label", "support", "predicted", "precision", "recall", "f1"],
            )
            write_confusion(out_dir / f"confusion_{model}_{prompt_id}.csv", metric_rows)
            plot_confusion(out_dir / confusion_name, metric_rows, f"{model}: Confusion Matrix ({prompt_id})")

        plot_prediction_distribution(
            out_dir / distribution_name,
            rows,
            f"True vs Predicted Style Distribution ({model}, {prompt_id})",
        )

        manifest.append(
            {
                "model": model,
                "prompt_id": prompt_id,
                "rows": len(rows),
                "single_label_rows": len(single_rows),
                "multilabel_rows": len(multilabel_rows),
                "compact_csv": compact_name,
                "full_csv": full_name,
                "multilabel_csv": multilabel_name,
                "per_class_csv": table_name if metric_rows else "",
                "confusion_png": confusion_name if metric_rows else "",
                "distribution_png": distribution_name,
            }
        )

    write_csv(
        out_dir / "detailed_outputs_manifest.csv",
        manifest,
        [
            "model",
            "prompt_id",
            "rows",
            "single_label_rows",
            "multilabel_rows",
            "compact_csv",
            "full_csv",
            "multilabel_csv",
            "per_class_csv",
            "confusion_png",
            "distribution_png",
        ],
    )

    print(f"detailed_rows={len(detailed_rows)} groups={len(groups)}")
    print(f"out_dir={out_dir}")
    for row in manifest:
        print(row)


if __name__ == "__main__":
    main()
