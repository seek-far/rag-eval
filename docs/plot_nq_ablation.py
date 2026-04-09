from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
RESULTS_PATH = ROOT / "results" / "results.jsonl"
OUTPUT_DIR = DOCS_DIR / "figures"
OUTPUT_PATH = OUTPUT_DIR / "nq_400_ablation.png"
MPLCONFIGDIR = DOCS_DIR / ".mplconfig"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


TARGETS = [
    ("semantic", "cross-encoder"),
    ("sentence", "cross-encoder"),
    ("fixed", "cross-encoder"),
    ("sentence", "none"),
    ("fixed", "none"),
]

LABELS = {
    ("semantic", "cross-encoder"): "Semantic + CE",
    ("sentence", "cross-encoder"): "Sentence + CE",
    ("fixed", "cross-encoder"): "Fixed + CE",
    ("sentence", "none"): "Sentence only",
    ("fixed", "none"): "Fixed only",
}

COLORS = {
    "cross-encoder": "#1f77b4",
    "none": "#9aa0a6",
}


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with RESULTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_nq_rows(rows: list[dict]) -> list[dict]:
    selected: dict[tuple[str, str], dict] = {}
    for row in rows:
        config = row.get("config", {})
        key = (config.get("chunk_strategy"), config.get("reranker"))
        if (
            config.get("dataset") == "nq"
            and row.get("n_samples") == 400
            and config.get("embed_model") == "BAAI/bge-large-en-v1.5"
            and config.get("retrieval_mode") == "hybrid"
            and key in TARGETS
        ):
            selected[key] = row
    missing = [key for key in TARGETS if key not in selected]
    if missing:
        raise SystemExit(f"Missing expected rows: {missing}")
    return [selected[key] for key in TARGETS]


def build_plot(rows: list[dict]) -> None:
    labels = [LABELS[(r["config"]["chunk_strategy"], r["config"]["reranker"])] for r in rows]
    mrr1 = [r["metrics"]["mrr@1"] for r in rows]
    recall10 = [r["metrics"]["recall@10"] for r in rows]
    colors = [COLORS[r["config"]["reranker"]] for r in rows]
    positions = list(range(len(rows)))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
    fig.suptitle("Natural Questions Ablation (400 samples)", fontsize=16, fontweight="bold")

    charts = [
        (axes[0], mrr1, "MRR@1", (0.84, 0.97)),
        (axes[1], recall10, "Recall@10", (0.97, 1.00)),
    ]

    for ax, values, title, xlim in charts:
        bars = ax.barh(positions, values, color=colors, edgecolor="none", height=0.68)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=11)
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
    MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    selected = select_nq_rows(rows)
    build_plot(selected)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
