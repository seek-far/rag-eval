"""
reporter.py — Saves per-run results and prints a cross-run comparison table.

Each run appends one JSON line to results/results.jsonl so you can
compare many experiments without re-running them.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from statistics import mean

logger = logging.getLogger(__name__)


def aggregate(per_sample: list[dict[str, float]]) -> dict[str, float]:
    if not per_sample:
        return {}
    keys = per_sample[0].keys()
    return {k: round(mean(r.get(k, 0.0) for r in per_sample), 4) for k in keys}


def save_run(cfg, per_sample: list[dict], agg: dict[str, float]) -> None:
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(cfg.results_dir) / "results.jsonl"

    row = {
        "run_name": cfg.run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "dataset": cfg.dataset,
            "split": cfg.split,
            "sample": cfg.sample,
            "chunk_strategy": cfg.chunk_strategy,
            "chunk_size": cfg.chunk_size,
            "semantic_threshold": cfg.semantic_threshold,
            "embed_model": cfg.embed_model,
            "embed_device": cfg.embed_device,
            "retrieval_mode": cfg.retrieval_mode,
            "retrieve_k": cfg.retrieve_k,
            "rrf_k": cfg.rrf_k,
            "dense_weight": cfg.dense_weight,
            "sparse_weight": cfg.sparse_weight,
            "reranker": cfg.reranker,
            "reranker_model": cfg.reranker_model,
            "rerank_top_k": cfg.rerank_top_k,
        },
        "n_samples": len(per_sample),
        "metrics": agg,
    }

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Results saved → %s", jsonl_path)
    _print_table(jsonl_path)


def _print_table(jsonl_path: Path) -> None:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return

    # Collect all metric keys that appear across runs
    metric_keys = []
    seen = set()
    for r in rows:
        for k in r.get("metrics", {}):
            if k not in seen:
                metric_keys.append(k)
                seen.add(k)

    cfg_cols = ["run_name", "dataset", "chunk_strategy",
                "embed_model", "retrieval_mode", "reranker"]
    header = cfg_cols + metric_keys

    # Column widths
    widths = {h: max(len(h), 6) for h in header}
    for r in rows:
        c = r["config"]
        vals = {
            "run_name": r.get("run_name", ""),
            "dataset": c.get("dataset", ""),
            "chunk_strategy": c.get("chunk_strategy", ""),
            "embed_model": c.get("embed_model", "").split("/")[-1],
            "retrieval_mode": c.get("retrieval_mode", ""),
            "reranker": c.get("reranker", ""),
        }
        for h in cfg_cols:
            widths[h] = max(widths[h], len(str(vals.get(h, ""))))
        for k in metric_keys:
            widths[k] = max(widths[k], len(f"{r['metrics'].get(k, 0):.4f}"))

    sep = "  "

    def fmt_row(cells: list[str]) -> str:
        return sep.join(str(c).ljust(widths[h]) for h, c in zip(header, cells))

    print("\n" + "─" * (sum(widths.values()) + len(sep) * (len(header) - 1)))
    print(fmt_row(header))
    print("─" * (sum(widths.values()) + len(sep) * (len(header) - 1)))

    for r in rows:
        c = r["config"]
        cfg_vals = [
            r.get("run_name", ""),
            c.get("dataset", ""),
            c.get("chunk_strategy", ""),
            c.get("embed_model", "").split("/")[-1],
            c.get("retrieval_mode", ""),
            c.get("reranker", ""),
        ]
        metric_vals = [
            f"{r['metrics'].get(k, 0):.4f}" for k in metric_keys
        ]
        print(fmt_row(cfg_vals + metric_vals))

    print("─" * (sum(widths.values()) + len(sep) * (len(header) - 1)) + "\n")
