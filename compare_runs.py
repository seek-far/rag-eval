"""
compare_runs.py — Pretty-print and optionally filter the results.jsonl table.

Usage:
    python compare_runs.py                        # show all runs
    python compare_runs.py --dataset nq           # filter by dataset
    python compare_runs.py --metric mrr@10        # sort by a metric
    python compare_runs.py --last 5               # show last N runs
    python compare_runs.py --csv                  # output CSV to stdout
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def load_rows(results_dir: str) -> list[dict]:
    path = Path(results_dir) / "results.jsonl"
    if not path.exists():
        print(f"No results file found at {path}")
        sys.exit(0)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten(row: dict) -> dict:
    c = row.get("config", {})
    flat = {
        "run_name":       row.get("run_name", ""),
        "timestamp":      row.get("timestamp", ""),
        "dataset":        c.get("dataset", ""),
        "split":          c.get("split", ""),
        "n_samples":      row.get("n_samples", ""),
        "chunk_strategy": c.get("chunk_strategy", ""),
        "chunk_size":     c.get("chunk_size", ""),
        "sem_threshold":  c.get("semantic_threshold", ""),
        "embed_model":    c.get("embed_model", "").split("/")[-1],
        "device":         c.get("embed_device", ""),
        "retrieval":      c.get("retrieval_mode", ""),
        "retrieve_k":     c.get("retrieve_k", ""),
        "rrf_k":          c.get("rrf_k", ""),
        "dense_w":        c.get("dense_weight", ""),
        "sparse_w":       c.get("sparse_weight", ""),
        "reranker":       c.get("reranker", ""),
        "rerank_model":   c.get("reranker_model", "").split("/")[-1],
        "rerank_top_k":   c.get("rerank_top_k", ""),
    }
    for k, v in row.get("metrics", {}).items():
        flat[k] = f"{v:.4f}"
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RAG evaluation runs")
    parser.add_argument("--results-dir", default=os.getenv("RESULTS_DIR", "./results"))
    parser.add_argument("--dataset", help="Filter by dataset name")
    parser.add_argument("--metric", help="Sort descending by this metric (e.g. mrr@10)")
    parser.add_argument("--last", type=int, default=0, help="Show last N runs")
    parser.add_argument("--csv", action="store_true", help="Output CSV to stdout")
    args = parser.parse_args()

    rows = load_rows(args.results_dir)

    if args.dataset:
        rows = [r for r in rows if r.get("config", {}).get("dataset") == args.dataset]

    if args.last:
        rows = rows[-args.last:]

    if args.metric:
        rows.sort(key=lambda r: r.get("metrics", {}).get(args.metric, 0), reverse=True)

    if not rows:
        print("No matching runs found.")
        return

    flat_rows = [flatten(r) for r in rows]

    # Collect all keys preserving order
    all_keys: list[str] = []
    seen: set[str] = set()
    for fr in flat_rows:
        for k in fr:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
        return

    # Pretty table
    widths = {k: len(k) for k in all_keys}
    for fr in flat_rows:
        for k in all_keys:
            widths[k] = max(widths[k], len(str(fr.get(k, ""))))

    sep = "  "
    total_w = sum(widths.values()) + len(sep) * (len(all_keys) - 1)

    def fmt(row: dict) -> str:
        return sep.join(str(row.get(k, "")).ljust(widths[k]) for k in all_keys)

    print("\n" + "─" * total_w)
    print(fmt({k: k for k in all_keys}))
    print("─" * total_w)
    for fr in flat_rows:
        print(fmt(fr))
    print("─" * total_w)
    print(f"  {len(flat_rows)} run(s) shown\n")


if __name__ == "__main__":
    main()
