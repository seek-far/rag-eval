from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt


DOCS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DOCS_DIR / "figures"
OUTPUT_PATH = OUTPUT_DIR / "scifact_300_ablation.png"

ROWS = [
    {
        "label": "Dense\n(no reranker)",
        "mrr1": 0.8100,
        "mrr10": 0.8646,
        "recall5": 0.9310,
        "group": "baseline",
    },
    {
        "label": "Dense + CE\n(MiniLM-L12)",
        "mrr1": 0.7600,
        "mrr10": 0.8219,
        "recall5": 0.9038,
        "group": "general_ce",
    },
    {
        "label": "Dense + CE\n(BGE large)",
        "mrr1": 0.7700,
        "mrr10": 0.8388,
        "recall5": 0.9236,
        "group": "general_ce",
    },
    {
        "label": "Dense + CE\n(BiomedBERT)",
        "mrr1": 0.8400,
        "mrr10": 0.8832,
        "recall5": 0.9367,
        "group": "domain_ce",
    },
    {
        "label": "Dense + CE\n(MedCPT)",
        "mrr1": 0.8467,
        "mrr10": 0.8914,
        "recall5": 0.9402,
        "group": "domain_ce",
    },
    {
        "label": "Dense + LLM CE\n(DeepSeek)",
        "mrr1": 0.8500,
        "mrr10": 0.8872,
        "recall5": 0.9243,
        "group": "llm_ce",
    },
    {
        "label": "Fixed score fusion\n(5-fold)",
        "mrr1": 0.8534,
        "mrr10": 0.8975,
        "recall5": 0.9500,
        "group": "fusion",
    },
    {
        "label": "Learned fusion\n(random forest, 5-fold)",
        "mrr1": 0.8600,
        "mrr10": 0.9059,
        "recall5": 0.9572,
        "group": "fusion",
    },
]

COLORS = {
    "baseline": "#4d6f7f",
    "general_ce": "#8a8f98",
    "domain_ce": "#2a9d8f",
    "llm_ce": "#7c5fb8",
    "fusion": "#d98324",
}


def build_plot() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.2), constrained_layout=True)
    fig.suptitle(
        "SciFact Ablation (300 samples): domain CEs and offline fusion",
        fontsize=16,
        fontweight="bold",
    )

    labels = [row["label"] for row in ROWS]
    colors = [COLORS[row["group"]] for row in ROWS]
    positions = list(range(len(ROWS)))
    baseline_mrr1 = ROWS[0]["mrr1"]

    charts = [
        (axes[0], [row["mrr1"] for row in ROWS], "MRR@1", (0.74, 0.93)),
        (axes[1], [row["mrr10"] for row in ROWS], "MRR@10", (0.81, 0.915)),
        (axes[2], [row["recall5"] for row in ROWS], "Recall@5", (0.90, 0.965)),
    ]

    for ax, values, title, xlim in charts:
        bars = ax.barh(positions, values, color=colors, edgecolor="none", height=0.68)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_yticks(positions)
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=10.2)
        ax.set_xlim(*xlim)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)

        for bar, value in zip(bars, values):
            label = f"{value:.4f}"
            if title == "MRR@1" and value != baseline_mrr1:
                label = f"{value:.4f} ({value - baseline_mrr1:+.4f})"
            ax.text(
                min(value + 0.001, xlim[1] - 0.0005),
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontsize=10,
                color="#222222",
            )

    axes[0].axvline(baseline_mrr1, color="#333333", linewidth=1, alpha=0.45)
    axes[0].text(
        baseline_mrr1 + 0.001,
        -0.62,
        "dense baseline",
        fontsize=9,
        color="#333333",
        ha="left",
        va="center",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    build_plot()
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
