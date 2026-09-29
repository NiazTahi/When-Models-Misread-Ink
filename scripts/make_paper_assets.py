#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUT = ROOT / "paper_assets"
LABELS = ["Kufic", "Naskh", "Thuluth", "Diwani", "Ruq'ah", "Nasta'liq"]
MODEL_ORDER = [
    "GPT-4o",
    "GPT-4.1",
    "GPT-5.5",
    "GPT-5.6 Terra",
    "Qwen2.5-VL-7B",
    "Qwen2.5-VL-32B",
    "Qwen3-VL-8B",
    "Qwen3-VL-32B",
    "InternVL3.5-8B",
    "InternVL3.5-38B",
    "MiniCPM-V 4.5",
    "Gemma 4 12B-IT",
]


def use_paper_font(base_size: int = 8) -> None:
    """Use a Times-style serif font for paper figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": base_size,
            "axes.titlesize": base_size + 1,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "legend.fontsize": base_size - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pct(x: str) -> float:
    return 100 * float(x)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_main_numeric_table(summary: list[dict]) -> None:
    rows = []
    for model in MODEL_ORDER:
        r = next(x for x in summary if x["model"] == model and x["prompt"] == "Numeric")
        rows.append(
            {
                "Group": r["model_group"],
                "Model": model,
                "Accuracy ↑": f"{pct(r['accuracy']):.1f}",
                "Macro-F1 ↑": f"{float(r['macro_f1']):.3f}",
                "Thuluth ratio ↓": f"{float(r['thuluth_ratio']):.2f}x",
                "Diwani recall ↑": f"{pct(r['diwani_recall']):.1f}",
            }
        )
    write_csv(OUT / "table_main_numeric_results.csv", rows, list(rows[0].keys()))


def select_prompt_effect_table(summary: list[dict]) -> None:
    by_key = {(r["model_group"], r["model"], r["prompt"]): r for r in summary}
    rows = []
    for model in MODEL_ORDER:
        group = next(r["model_group"] for r in summary if r["model"] == model)
        n = by_key[(group, model, "Numeric")]
        v = by_key[(group, model, "Visual-cue")]
        d = by_key[(group, model, "Diwani-first")]
        rows.append(
            {
                "Group": group,
                "Model": model,
                "Visual-cue Δ accuracy ↑": f"{pct(v['accuracy']) - pct(n['accuracy']):+.1f}",
                "Diwani-first Δ accuracy ↑": f"{pct(d['accuracy']) - pct(v['accuracy']):+.1f}",
                "Diwani-first Δ Diwani recall ↑": f"{pct(d['diwani_recall']) - pct(v['diwani_recall']):+.1f}",
            }
        )
    write_csv(OUT / "table_prompt_effects.csv", rows, list(rows[0].keys()))


def plot_prompt_effects(summary: list[dict]) -> None:
    all_predictions = read_csv(ANALYSIS / "unified_all_predictions.csv")

    def exact_mcnemar_p(model: str, prompt_a: str, prompt_b: str) -> float:
        a = {
            (r["filename"], r["true_label"]): r["pred_label"] == r["true_label"]
            for r in all_predictions
            if r["model"] == model and r["prompt"] == prompt_a
        }
        b = {
            (r["filename"], r["true_label"]): r["pred_label"] == r["true_label"]
            for r in all_predictions
            if r["model"] == model and r["prompt"] == prompt_b
        }
        better = sum((not a[k]) and b[k] for k in a)
        worse = sum(a[k] and (not b[k]) for k in a)
        n = better + worse
        if n == 0:
            return 1.0
        tail = sum(math.comb(n, i) for i in range(min(better, worse) + 1)) / (2**n)
        return min(1.0, 2 * tail)

    by_key = {(r["model_group"], r["model"], r["prompt"]): r for r in summary}
    rows = []
    for model in MODEL_ORDER:
        group = next(r["model_group"] for r in summary if r["model"] == model)
        n = by_key[(group, model, "Numeric")]
        v = by_key[(group, model, "Visual-cue")]
        d = by_key[(group, model, "Diwani-first")]
        rows.append(
            {
                "model": model,
                "group": group,
                "acc_delta": pct(v["accuracy"]) - pct(n["accuracy"]),
                "diwani_delta": pct(d["diwani_recall"]) - pct(v["diwani_recall"]),
                "visual_p": exact_mcnemar_p(model, "Numeric", "Visual-cue"),
            }
        )

    use_paper_font(8)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4), sharey=True)
    y = np.arange(len(rows))
    help_color = "#2A9D8F"
    hurt_color = "#E76F51"

    panels = [
        (axes[0], "acc_delta", "(a) Visual cues", "Accuracy change vs. numeric (pp)"),
        (axes[1], "diwani_delta", "(b) Diwani-first", "Diwani recall change vs. visual-cue (pp)"),
    ]
    for ax, key, title, xlabel in panels:
        vals = [r[key] for r in rows]
        colors = [help_color if v >= 0 else hurt_color for v in vals]
        bars = ax.barh(y, vals, color=colors, alpha=0.92)
        if key == "acc_delta":
            for bar, r in zip(bars, rows):
                if r["visual_p"] >= 0.05:
                    bar.set_alpha(0.38)
                    bar.set_hatch("//")
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
        lim = max(abs(min(vals)), abs(max(vals))) + 5
        ax.set_xlim(-lim, lim)
        for yi, v, r in zip(y, vals, rows):
            ha = "left" if v >= 0 else "right"
            offset = 0.7 if v >= 0 else -0.7
            label = f"{v:+.1f}"
            if key == "acc_delta" and r["visual_p"] >= 0.05:
                label = f"{label} n.s."
            ax.text(v + offset, yi, label, va="center", ha=ha, fontsize=6.5, color="#333333")

    axes[0].set_yticks(y, [r["model"] for r in rows])
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label="Improves target metric", markerfacecolor=help_color, markersize=7),
        plt.Line2D([0], [0], marker="s", color="w", label="Hurts target metric", markerfacecolor=hurt_color, markersize=7),
        plt.Rectangle((0, 0), 1, 1, facecolor="#9CA3AF", alpha=0.38, hatch="//", label="Visual-cue change not significant"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.02), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.86, bottom=0.23, wspace=0.20)
    fig.savefig(OUT / "figure_prompt_effects.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "figure_prompt_effects.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_bias_visibility(summary: list[dict]) -> None:
    rows = []
    for model in MODEL_ORDER:
        r = next(x for x in summary if x["model"] == model and x["prompt"] == "Numeric")
        rows.append(
            {
                "model": model,
                "group": r["model_group"],
                "thuluth_ratio": float(r["thuluth_ratio"]),
                "diwani_recall": pct(r["diwani_recall"]),
                "accuracy": pct(r["accuracy"]),
            }
        )

    use_paper_font(8)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.4, 3.7))

    markers = {"Closed/API": "o", "Open-weight": "D"}
    colors = {"Closed/API": "#4B5563", "Open-weight": "#2A9D8F"}
    for group in ["Closed/API", "Open-weight"]:
        subset = [r for r in rows if r["group"] == group]
        ax.scatter(
            [r["thuluth_ratio"] for r in subset],
            [r["diwani_recall"] for r in subset],
            s=[34 + r["accuracy"] * 1.2 for r in subset],
            marker=markers[group],
            color=colors[group],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
            label=group,
            zorder=3,
        )

    label_offsets = {
        "GPT-5.5": (0.04, 2.4),
        "GPT-5.6 Terra": (0.05, -3.5),
        "GPT-4.1": (0.04, -4.0),
        "InternVL3.5-38B": (0.04, 2.1),
        "InternVL3.5-8B": (0.04, -2.2),
        "Qwen2.5-VL-7B": (-0.52, 2.0),
        "Qwen3-VL-32B": (0.04, -2.4),
        "Gemma 4 12B-IT": (0.04, -2.5),
    }
    for r in rows:
        dx, dy = label_offsets.get(r["model"], (0.04, 1.2))
        ax.text(r["thuluth_ratio"] + dx, r["diwani_recall"] + dy, r["model"], fontsize=6.5)

    ax.axvline(1.0, color="#9CA3AF", linestyle="--", linewidth=0.8)
    ax.axhspan(0, 15, color="#F3F4F6", zorder=0)
    ax.text(1.03, 59.5, "ideal Thuluth ratio", color="#6B7280", fontsize=7, rotation=90, va="top")
    ax.text(3.62, 7.5, "low Diwani visibility", color="#6B7280", fontsize=7, ha="right", va="center")
    ax.set_xlim(0.45, 3.72)
    ax.set_ylim(-2, 64)
    ax.set_xlabel("Thuluth prediction ratio (1.0 = balanced)")
    ax.set_ylabel("Diwani recall (%)")
    ax.set_title("Dominant-label absorption versus Diwani visibility")
    ax.grid(color="#E5E7EB", linewidth=0.6, zorder=0)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.savefig(OUT / "figure_bias_visibility.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "figure_bias_visibility.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_thuluth_diwani_failure(summary: list[dict], confusions: list[dict]) -> None:
    summary_by_model = {r["model"]: r for r in summary if r["prompt"] == "Numeric"}
    rows = []
    source_labels = ["Diwani", "Naskh", "Nasta'liq", "Ruq'ah", "Kufic"]
    for model in MODEL_ORDER:
        rs = [r for r in confusions if r["prompt"] == "Numeric" and r["model"] == model]
        false_sources = {}
        for label in source_labels:
            row = next(r for r in rs if r["true_label"] == label)
            false_sources[label] = int(row["Thuluth"])
        rows.append(
            {
                "model": model,
                "group": summary_by_model[model]["model_group"],
                "diwani_recall": pct(summary_by_model[model]["diwani_recall"]),
                "false_sources": false_sources,
            }
        )

    use_paper_font(8)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.5), sharey=True, gridspec_kw={"width_ratios": [1.16, 0.84]})
    y = np.arange(len(rows))
    source_colors = {
        "Diwani": "#D95F02",
        "Naskh": "#7570B3",
        "Nasta'liq": "#1B9E77",
        "Ruq'ah": "#E6AB02",
        "Kufic": "#A6761D",
    }

    left = np.zeros(len(rows))
    for label in source_labels:
        vals = np.array([r["false_sources"][label] for r in rows])
        axes[0].barh(y, vals, left=left, color=source_colors[label], label=label, alpha=0.92)
        left += vals
    axes[0].set_yticks(y, [r["model"] for r in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("False Thuluth predictions (count)")
    axes[0].set_title("(a) What gets absorbed into Thuluth?")
    axes[0].grid(axis="x", color="#E5E7EB", linewidth=0.6)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.50, -0.11), frameon=False, fontsize=6.2, ncol=5, handlelength=1.0, columnspacing=0.75)

    recall_vals = [r["diwani_recall"] for r in rows]
    recall_colors = ["#4B5563" if r["group"] == "Closed/API" else "#2A9D8F" for r in rows]
    axes[1].barh(y, recall_vals, color=recall_colors, alpha=0.92)
    axes[1].set_xlabel("Diwani recall (%)")
    axes[1].set_title("(b) Diwani remains weak")
    axes[1].set_xlim(0, 62)
    axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.6)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    for yi, val in zip(y, recall_vals):
        axes[1].text(val + 1.0, yi, f"{val:.1f}", va="center", fontsize=6.4)
    group_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label="Closed/API", markerfacecolor="#4B5563", markersize=7),
        plt.Line2D([0], [0], marker="s", color="w", label="Open-weight", markerfacecolor="#2A9D8F", markersize=7),
    ]
    axes[1].legend(handles=group_handles, loc="lower right", frameon=False, fontsize=6.4)

    fig.subplots_adjust(left=0.18, right=0.99, top=0.88, bottom=0.22, wspace=0.08)
    fig.savefig(OUT / "figure_thuluth_diwani_failure.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "figure_thuluth_diwani_failure.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_prompt_radars(per_class: list[dict]) -> None:
    selected_models = [
        "GPT-4.1",
        "GPT-5.5",
        "GPT-5.6 Terra",
        "Qwen3-VL-8B",
        "Qwen3-VL-32B",
        "InternVL3.5-38B",
    ]
    prompts = ["Numeric", "Visual-cue", "Diwani-first"]
    prompt_colors = {"Numeric": "#4B5563", "Visual-cue": "#2A9D8F", "Diwani-first": "#E76F51"}
    by_key = {(r["model"], r["prompt"], r["label"]): r for r in per_class}
    angles = np.linspace(0, 2 * np.pi, len(LABELS), endpoint=False).tolist()
    angles += angles[:1]

    use_paper_font(7)
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.8), subplot_kw={"projection": "polar"})
    for ax, model in zip(axes.ravel(), selected_models):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1], LABELS, fontsize=6.2)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=5.8, color="#6B7280")
        ax.grid(color="#D1D5DB", linewidth=0.55)
        for prompt in prompts:
            vals = [pct(by_key[(model, prompt, label)]["recall"]) for label in LABELS]
            vals += vals[:1]
            ax.plot(angles, vals, color=prompt_colors[prompt], linewidth=1.1, label=prompt)
            ax.fill(angles, vals, color=prompt_colors[prompt], alpha=0.08)
        ax.set_title(model, fontsize=8, fontweight="bold", pad=9)

    handles = [
        plt.Line2D([0], [0], color=prompt_colors[p], linewidth=1.4, label=p)
        for p in prompts
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.10, wspace=0.22, hspace=0.34)
    fig.savefig(OUT / "figure_prompt_radars.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "figure_prompt_radars.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_pair(confusions: list[dict]) -> None:
    picks = [("Closed/API", "GPT-4.1", "Numeric"), ("Open-weight", "Qwen3-VL-32B", "Numeric")]
    use_paper_font(8)
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.3), constrained_layout=True)
    vmax = 0
    matrices = []
    for group, model, prompt in picks:
        rows = [r for r in confusions if r["model_group"] == group and r["model"] == model and r["prompt"] == prompt]
        mat = np.array([[int(row[label]) for label in LABELS] for row in rows])
        matrices.append((group, model, prompt, mat))
        vmax = max(vmax, mat.max())
    for ax, (_, model, prompt, mat) in zip(axes, matrices):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_title(f"{model}, {prompt}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color="white" if val > vmax * 0.55 else "black")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.76, label="count")
    fig.savefig(OUT / "figure_representative_confusions.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure_representative_confusions.pdf", bbox_inches="tight")
    plt.close(fig)


def make_dataset_examples() -> None:
    use_paper_font(9)
    manifest = read_csv(ROOT / "dataset" / "merged_manifest.csv")
    examples = {}
    for label in LABELS:
        row = next(r for r in manifest if r["style"] == label)
        examples[label] = ROOT / "dataset" / "images" / row["filename"]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)
    for ax, label in zip(axes.ravel(), LABELS):
        img = Image.open(examples[label]).convert("RGB")
        ax.imshow(img)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axis("off")
    fig.savefig(OUT / "figure_dataset_examples.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figure_dataset_examples.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = read_csv(ANALYSIS / "unified_summary_metrics.csv")
    confusions = read_csv(ANALYSIS / "unified_confusion_counts.csv")
    per_class = read_csv(ANALYSIS / "unified_per_class_metrics.csv")
    select_main_numeric_table(summary)
    select_prompt_effect_table(summary)
    plot_prompt_effects(summary)
    plot_bias_visibility(summary)
    plot_thuluth_diwani_failure(summary, confusions)
    plot_prompt_radars(per_class)
    plot_confusion_pair(confusions)
    make_dataset_examples()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
