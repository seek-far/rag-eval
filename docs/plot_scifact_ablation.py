from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt


DOCS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DOCS_DIR / "figures"
OUTPUT_PATH = OUTPUT_DIR / "scifact_400_ablation.png"

ROWS = [
    {
        "label": "Sentence + hybrid + CE\n(MiniLM)",
        "mrr10": 0.8345,
        "mrr1": 0.7800,
        "recall10": 0.9027,
        "group": "hybrid_ce",
    },
    {
        "label": "Semantic + hybrid + CE\n(MiniLM)",
        "mrr10": 0.8344,
        "mrr1": 0.7800,
        "recall10": 0.9027,
        "group": "hybrid_ce",
    },
    {
        "label": "Semantic + hybrid + CE\n(BGE reranker large)",
        "mrr10": 0.8339,
        "mrr1": 0.7667,
        "recall10": 0.9236,
        "group": "hybrid_ce_bge",
    },
    {
        "label": "Sentence + dense\n(no reranker)",
        "mrr10": 0.8646,
        "mrr1": 0.8100,
        "recall10": 0.9550,
        "group": "dense_none",
    },
]

COLORS = {
    "hybrid_ce": "#1f77b4",
    "hybrid_ce_bge": "#4c78a8",
    "dense_none": "#2a9d8f",
}


def build_plot() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), constrained_layout=True)
    fig.suptitle("SciFact Ablation (400 samples)", fontsize=16, fontweight="bold")

    labels = [row["label"] for row in ROWS]
    colors = [COLORS[row["group"]] for row in ROWS]
    positions = list(range(len(ROWS)))

    charts = [
        (axes[0], [row["mrr1"] for row in ROWS], "MRR@1", (0.75, 0.82)),
        (axes[1], [row["recall10"] for row in ROWS], "Recall@10", (0.89, 0.965)),
    ]

    for ax, values, title, xlim in charts:
        bars = ax.barh(positions, values, color=colors, edgecolor="none", height=0.68)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=10.5)
        ax.set_xlim(*xlim)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)

        for bar, value in zip(bars, values):
            ax.text(
                min(value + 0.001, xlim[1] - 0.0005),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}",
                va="center",
                ha="left",
                fontsize=10,
                color="#222222",
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    build_plot()
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
