"""
case_analysis.py — Analyse success and failure cases for a specific run config.

Re-runs the evaluation pipeline (using cached chunks/embeddings) and classifies
each query as success or failure based on hit@K. Displays detailed breakdowns.

Usage:
    python -m utils.case_analysis                              # use .env config
    python -m utils.case_analysis --metric hit@1               # success = hit@1==1
    python -m utils.case_analysis --show failures              # only show failures
    python -m utils.case_analysis --show successes --top 5     # top 5 successes
    python -m utils.case_analysis --export failures.jsonl      # export to file
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from dataloader.loader import load_eval_samples
from indexing.chunker import build_chunks, create_chunker
from indexing.embedder import Embedder
from indexing.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever
from reranking.cross_encoder import Reranker
from eval.retrieval_metrics import (
    compute_retrieval_metrics, compute_context_metrics,
    mrr_at_k, hit_at_k, recall_at_k,
)


def _trunc(text: str, n: int = 120) -> str:
    return text[:n] + "..." if len(text) > n else text


def _run_pipeline(cfg):
    """Run the full pipeline and return per-sample details."""
    samples = load_eval_samples(cfg.dataset, cfg.split, cfg.total_samples)
    samples = samples[:cfg.total_samples]

    # Build chunks
    cfg.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_map: dict[str, dict] = {}
    chunker = None

    for batch_start, batch_end in cfg.sample_batches:
        cache_path = cfg.chunk_cache_dir / f"batch_{batch_start}_{batch_end}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                batch_chunks = pickle.load(f)
        else:
            batch_samples = samples[batch_start:batch_end]
            if chunker is None:
                chunker = create_chunker(cfg)
            all_chunks = []
            for sample in tqdm(batch_samples, desc=f"Chunking [{batch_start}:{batch_end}]"):
                all_chunks.extend(build_chunks(sample.documents, cfg, chunker=chunker))
            batch_chunks = list({c["chunk_id"]: c for c in all_chunks}.values())
            if batch_chunks:
                with open(cache_path, "wb") as f:
                    pickle.dump(batch_chunks, f)

        for c in batch_chunks:
            chunk_map.setdefault(c["chunk_id"], c)

    unique_chunks = list(chunk_map.values())

    # Build indices
    embedder = Embedder(cfg)
    embedder.build_index(unique_chunks)
    bm25 = BM25Index(unique_chunks)
    retriever = HybridRetriever(embedder, bm25, cfg)
    reranker = Reranker(cfg) if cfg.reranker != "none" else None

    # Evaluate each sample and collect details
    results = []
    for sample in tqdm(samples, desc="Evaluating", unit="query"):
        candidates = retriever.retrieve(sample.query, chunk_map)

        if reranker is not None:
            candidates = reranker.rerank(sample.query, candidates)
            candidates = candidates[:cfg.rerank_top_k]

        # Deduplicate
        seen = set()
        retrieved_ids = []
        for c in candidates:
            if c["doc_id"] not in seen:
                seen.add(c["doc_id"])
                retrieved_ids.append(c["doc_id"])

        # Metrics
        metrics = {}
        if sample.relevant_doc_ids:
            metrics.update(compute_retrieval_metrics(
                retrieved_ids, sample.relevant_doc_ids, cfg.k_values, cfg.metrics))

        if sample.answer_spans and candidates:
            chunk_texts = [c["text"] for c in candidates]
            metrics.update(compute_context_metrics(
                chunk_texts, sample.answer_spans, cfg.k_values))

        results.append({
            "sample": sample,
            "candidates": candidates,
            "retrieved_ids": retrieved_ids,
            "metrics": metrics,
        })

    return results


def _print_case(r: dict, verbose: bool = True) -> None:
    sample = r["sample"]
    metrics = r["metrics"]
    print(f"\n  Sample id={sample.id}")
    print(f"  Query: {sample.query}")
    print(f"  Relevant: {sample.relevant_doc_ids}")
    print(f"  Retrieved: {r['retrieved_ids'][:10]}")

    # Show key metrics
    metric_strs = [f"{k}={v:.3f}" for k, v in sorted(metrics.items())
                   if any(k.startswith(m) for m in ("mrr", "hit", "recall", "ndcg"))]
    print(f"  Metrics: {', '.join(metric_strs[:12])}")

    if verbose and r["candidates"]:
        print(f"  Top candidates:")
        for i, c in enumerate(r["candidates"][:5]):
            is_rel = c["doc_id"] in sample.relevant_doc_ids
            tag = " [REL]" if is_rel else ""
            scores = []
            if "retrieval_score" in c:
                scores.append(f"ret={c['retrieval_score']:.4f}")
            if "rerank_score" in c:
                scores.append(f"rerank={c['rerank_score']:.4f}")
            print(f"    {i+1}. chunk={c['chunk_id']}  doc={c['doc_id']}{tag}  {' '.join(scores)}")
            print(f"       {_trunc(c['text'], 150)}")

    if verbose and sample.reference_answer:
        print(f"  Reference: {_trunc(sample.reference_answer, 200)}")
    if verbose and sample.answer_spans:
        print(f"  Answer spans: {sample.answer_spans[:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Success/failure case analysis")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--metric", default="hit@1",
                        help="Metric to classify success/failure (default: hit@1)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Score >= threshold = success (default: 1.0)")
    parser.add_argument("--show", choices=["all", "failures", "successes", "summary"],
                        default="all", help="What to display")
    parser.add_argument("--top", type=int, default=0,
                        help="Show only top N cases (0=all)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Show candidate details")
    parser.add_argument("--brief", action="store_true",
                        help="Hide candidate details")
    parser.add_argument("--export", default=None,
                        help="Export results to JSONL file")
    args = parser.parse_args()

    cfg = Config()
    if args.dataset:
        cfg.dataset = args.dataset
    if args.split:
        cfg.split = args.split

    print(f"\nRunning pipeline: {cfg.dataset}/{cfg.split}, "
          f"chunk={cfg.chunk_strategy}, embed={cfg.embed_model}, "
          f"retrieval={cfg.retrieval_mode}, reranker={cfg.reranker}")
    print(f"Success criterion: {args.metric} >= {args.threshold}\n")

    results = _run_pipeline(cfg)

    # Classify
    successes = []
    failures = []
    for r in results:
        score = r["metrics"].get(args.metric, 0.0)
        if score >= args.threshold:
            successes.append(r)
        else:
            failures.append(r)

    # Sort failures by metric ascending (worst first), successes by metric descending
    failures.sort(key=lambda r: r["metrics"].get(args.metric, 0.0))
    successes.sort(key=lambda r: r["metrics"].get(args.metric, 0.0), reverse=True)

    verbose = not args.brief

    # Summary
    print(f"\n{'='*70}")
    print(f"  Results: {len(successes)} successes, {len(failures)} failures "
          f"out of {len(results)} samples")
    print(f"  Success rate: {len(successes)/len(results)*100:.1f}%")
    print(f"{'='*70}")

    if args.show == "summary":
        # Show metric distribution
        for metric_name in sorted(set(k for r in results for k in r["metrics"])):
            vals = [r["metrics"].get(metric_name, 0.0) for r in results]
            avg = sum(vals) / len(vals) if vals else 0
            print(f"  {metric_name}: avg={avg:.4f}  "
                  f"min={min(vals):.4f}  max={max(vals):.4f}")
        return

    if args.show in ("all", "failures"):
        n_show = args.top if args.top > 0 else len(failures)
        print(f"\n{'─'*70}")
        print(f"  FAILURE CASES ({min(n_show, len(failures))} of {len(failures)})")
        print(f"{'─'*70}")
        for r in failures[:n_show]:
            _print_case(r, verbose=verbose)

    if args.show in ("all", "successes"):
        n_show = args.top if args.top > 0 else len(successes)
        print(f"\n{'─'*70}")
        print(f"  SUCCESS CASES ({min(n_show, len(successes))} of {len(successes)})")
        print(f"{'─'*70}")
        for r in successes[:n_show]:
            _print_case(r, verbose=verbose)

    # Export — respect --show filter
    if args.export:
        if args.show == "failures":
            export_list = failures
        elif args.show == "successes":
            export_list = successes
        else:
            export_list = results

        if args.top > 0:
            export_list = export_list[:args.top]

        export_data = []
        for r in export_list:
            score = r["metrics"].get(args.metric, 0.0)
            export_data.append({
                "sample_id": r["sample"].id,
                "query": r["sample"].query,
                "relevant_doc_ids": r["sample"].relevant_doc_ids,
                "retrieved_ids": r["retrieved_ids"],
                "reference_answer": r["sample"].reference_answer,
                "metrics": r["metrics"],
                "success": score >= args.threshold,
                "top_chunks": [
                    {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                     "text": c["text"][:300],
                     "retrieval_score": c.get("retrieval_score"),
                     "rerank_score": c.get("rerank_score")}
                    for c in r["candidates"][:5]
                ],
            })
        with open(args.export, "w", encoding="utf-8") as f:
            for d in export_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\nExported {len(export_data)} results to {args.export}")


if __name__ == "__main__":
    main()
