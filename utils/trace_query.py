"""
trace_query.py — Trace the full retrieval process for a single query.

Shows every stage of the pipeline with scores and rankings:
  1. Document & chunk info
  2. Dense retrieval scores
  3. Sparse (BM25) retrieval scores
  4. Hybrid RRF fusion
  5. Reranking scores
  6. Final deduplication & metric evaluation

Usage:
    python -m utils.trace_query --sample-idx 0       # trace first sample
    python -m utils.trace_query --sample-id "q123"   # trace by sample ID
    python -m utils.trace_query --query "who is ..."  # trace a custom query
    python -m utils.trace_query --sample-idx 5 --top 10  # show top 10 at each stage
"""
from __future__ import annotations

import argparse
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
from retrieval.hybrid import _rrf
from reranking.cross_encoder import Reranker
from eval.retrieval_metrics import compute_retrieval_metrics, compute_context_metrics


def _trunc(text: str, n: int = 120) -> str:
    return text[:n] + "..." if len(text) > n else text


def _rel_tag(doc_id: str, relevant_ids: list[str]) -> str:
    return " [REL]" if doc_id in relevant_ids else ""


def _build_pipeline(cfg, samples):
    """Build chunks + indices, return (chunk_map, embedder, bm25, reranker)."""
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

    embedder = Embedder(cfg)
    embedder.build_index(unique_chunks)
    bm25 = BM25Index(unique_chunks)
    reranker = Reranker(cfg) if cfg.reranker != "none" else None

    return chunk_map, embedder, bm25, reranker


def trace(query: str, sample, chunk_map: dict, embedder: Embedder,
          bm25, reranker, cfg, top_k: int = 10) -> None:
    """Trace and print every retrieval stage."""
    relevant_ids = sample.relevant_doc_ids if sample else []
    k = cfg.retrieve_k

    # ── Stage 0: Query info ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  QUERY TRACE")
    print(f"{'='*80}")
    print(f"  Query: {query}")
    if sample:
        print(f"  Sample ID: {sample.id}")
        print(f"  Relevant doc IDs: {relevant_ids}")
        if sample.reference_answer:
            print(f"  Reference answer: {_trunc(sample.reference_answer, 200)}")
        if sample.answer_spans:
            print(f"  Answer spans: {sample.answer_spans[:3]}")

    # Count chunks belonging to relevant docs
    if relevant_ids:
        rel_chunks = [cid for cid, c in chunk_map.items()
                      if c["doc_id"] in relevant_ids]
        print(f"  Relevant doc chunks in index: {len(rel_chunks)}")
        for rc_id in rel_chunks[:5]:
            c = chunk_map[rc_id]
            print(f"    {rc_id}: {_trunc(c['text'], 100)}")
        if len(rel_chunks) > 5:
            print(f"    ... and {len(rel_chunks) - 5} more")

    # ── Stage 1: Dense retrieval ─────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  STAGE 1: Dense Retrieval (top {top_k} of {k})")
    print(f"{'─'*80}")

    dense_results = embedder.search(query, k)
    for rank, (chunk_id, score) in enumerate(dense_results[:top_k]):
        c = chunk_map.get(chunk_id, {})
        doc_id = c.get("doc_id", "?")
        tag = _rel_tag(doc_id, relevant_ids)
        print(f"  {rank+1:>3}. score={score:.6f}  chunk={chunk_id}  "
              f"doc={doc_id}{tag}")
        print(f"       {_trunc(c.get('text', ''), 120)}")

    # Where are relevant docs in dense results?
    if relevant_ids:
        _print_relevant_positions("Dense", dense_results, chunk_map, relevant_ids)

    # ── Stage 2: Sparse (BM25) retrieval ─────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  STAGE 2: Sparse / BM25 Retrieval (top {top_k} of {k})")
    print(f"{'─'*80}")

    sparse_results = bm25.search(query, k)
    for rank, (chunk_id, score) in enumerate(sparse_results[:top_k]):
        c = chunk_map.get(chunk_id, {})
        doc_id = c.get("doc_id", "?")
        tag = _rel_tag(doc_id, relevant_ids)
        print(f"  {rank+1:>3}. score={score:.4f}  chunk={chunk_id}  "
              f"doc={doc_id}{tag}")
        print(f"       {_trunc(c.get('text', ''), 120)}")

    if relevant_ids:
        _print_relevant_positions("BM25", sparse_results, chunk_map, relevant_ids)

    # ── Stage 3: Hybrid RRF ──────────────────────────────────────────────
    if cfg.retrieval_mode == "hybrid":
        print(f"\n{'─'*80}")
        print(f"  STAGE 3: Hybrid RRF Fusion (k={cfg.rrf_k}, "
              f"dense_w={cfg.dense_weight}, sparse_w={cfg.sparse_weight})")
        print(f"{'─'*80}")

        rrf_results = _rrf(dense_results, sparse_results,
                           k=cfg.rrf_k, w_dense=cfg.dense_weight,
                           w_sparse=cfg.sparse_weight)

        for rank, (chunk_id, score) in enumerate(rrf_results[:top_k]):
            c = chunk_map.get(chunk_id, {})
            doc_id = c.get("doc_id", "?")
            tag = _rel_tag(doc_id, relevant_ids)

            # Show component ranks
            dense_rank = _find_rank(chunk_id, dense_results)
            sparse_rank = _find_rank(chunk_id, sparse_results)
            print(f"  {rank+1:>3}. rrf={score:.6f}  chunk={chunk_id}  "
                  f"doc={doc_id}{tag}  (dense_rank={dense_rank}, sparse_rank={sparse_rank})")
            print(f"       {_trunc(c.get('text', ''), 120)}")

        if relevant_ids:
            _print_relevant_positions("RRF", rrf_results, chunk_map, relevant_ids)

        candidates_ranked = rrf_results[:k]
    elif cfg.retrieval_mode == "dense":
        candidates_ranked = dense_results[:k]
    else:
        candidates_ranked = sparse_results[:k]

    # Build candidate chunk list
    candidates = []
    for chunk_id, score in candidates_ranked:
        if chunk_id in chunk_map:
            c = dict(chunk_map[chunk_id])
            c["retrieval_score"] = score
            candidates.append(c)

    # ── Stage 4: Reranking ───────────────────────────────────────────────
    if reranker is not None:
        print(f"\n{'─'*80}")
        print(f"  STAGE 4: Cross-Encoder Reranking (model={cfg.reranker_model})")
        print(f"{'─'*80}")

        candidates = reranker.rerank(query, candidates)
        candidates = candidates[:cfg.rerank_top_k]

        for rank, c in enumerate(candidates):
            tag = _rel_tag(c["doc_id"], relevant_ids)
            ret_score = c.get("retrieval_score", 0)
            rerank_score = c.get("rerank_score", 0)
            print(f"  {rank+1:>3}. rerank={rerank_score:.6f}  "
                  f"retrieval={ret_score:.6f}  chunk={c['chunk_id']}  "
                  f"doc={c['doc_id']}{tag}")
            print(f"       {_trunc(c['text'], 120)}")
    else:
        print(f"\n{'─'*80}")
        print(f"  STAGE 4: Reranking — SKIPPED (reranker=none)")
        print(f"{'─'*80}")

    # ── Stage 5: Deduplication & final result ────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  STAGE 5: Deduplication & Final Ranking")
    print(f"{'─'*80}")

    seen = set()
    retrieved_ids = []
    for c in candidates:
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            retrieved_ids.append(c["doc_id"])

    for rank, doc_id in enumerate(retrieved_ids):
        tag = _rel_tag(doc_id, relevant_ids)
        print(f"  {rank+1:>3}. doc={doc_id}{tag}")

    # ── Stage 6: Metrics ─────────────────────────────────────────────────
    if sample and sample.relevant_doc_ids:
        print(f"\n{'─'*80}")
        print(f"  STAGE 6: Metrics")
        print(f"{'─'*80}")

        metrics = compute_retrieval_metrics(
            retrieved_ids, sample.relevant_doc_ids,
            cfg.k_values, cfg.metrics)

        if sample.answer_spans and candidates:
            chunk_texts = [c["text"] for c in candidates]
            metrics.update(compute_context_metrics(
                chunk_texts, sample.answer_spans, cfg.k_values))

        for k_name in sorted(metrics):
            v = metrics[k_name]
            status = "PASS" if v >= 1.0 else ("PARTIAL" if v > 0 else "FAIL")
            print(f"  {k_name:>20s} = {v:.4f}  [{status}]")

    print(f"\n{'='*80}\n")


def _find_rank(chunk_id: str, results: list[tuple[str, float]]) -> str:
    for i, (cid, _) in enumerate(results):
        if cid == chunk_id:
            return str(i + 1)
    return "-"


def _print_relevant_positions(stage: str, results: list[tuple[str, float]],
                               chunk_map: dict, relevant_ids: list[str]) -> None:
    """Show where relevant doc chunks appear in results."""
    found = {}
    for rank, (chunk_id, score) in enumerate(results):
        doc_id = chunk_map.get(chunk_id, {}).get("doc_id", "?")
        if doc_id in relevant_ids and doc_id not in found:
            found[doc_id] = (rank + 1, chunk_id, score)

    missing = [rid for rid in relevant_ids if rid not in found]
    if found:
        parts = [f"{did}@rank{info[0]}(score={info[2]:.4f})"
                 for did, info in sorted(found.items(), key=lambda x: x[1][0])]
        print(f"  >> {stage} relevant doc positions: {', '.join(parts)}")
    if missing:
        print(f"  >> {stage} relevant docs NOT in top-{len(results)}: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace retrieval for a single query")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Index of the sample to trace (default: 0)")
    parser.add_argument("--sample-id", default=None,
                        help="Trace a sample by its ID")
    parser.add_argument("--query", default=None,
                        help="Trace a custom query (no ground truth)")
    parser.add_argument("--top", type=int, default=10,
                        help="Show top N results at each stage (default: 10)")
    args = parser.parse_args()

    cfg = Config()
    if args.dataset:
        cfg.dataset = args.dataset
    if args.split:
        cfg.split = args.split

    print(f"\nConfig: {cfg.dataset}/{cfg.split}, chunk={cfg.chunk_strategy}, "
          f"embed={cfg.embed_model}")
    print(f"  retrieval={cfg.retrieval_mode}, reranker={cfg.reranker}")

    samples = load_eval_samples(cfg.dataset, cfg.split, cfg.total_samples)
    samples = samples[:cfg.total_samples]

    print(f"Building pipeline ({len(samples)} samples)...")
    chunk_map, embedder, bm25, reranker_obj = _build_pipeline(cfg, samples)

    if args.query:
        # Custom query — no ground truth sample
        from dataloader.schema import EvalSample
        dummy = EvalSample(id="custom", query=args.query, documents=[],
                           relevant_doc_ids=[], reference_answer="")
        trace(args.query, dummy, chunk_map, embedder, bm25, reranker_obj,
              cfg, top_k=args.top)
    else:
        # Find the target sample
        if args.sample_id:
            matches = [s for s in samples if s.id == args.sample_id]
            if not matches:
                print(f"Sample ID '{args.sample_id}' not found.")
                return
            sample = matches[0]
        else:
            if args.sample_idx >= len(samples):
                print(f"Sample index {args.sample_idx} out of range (0-{len(samples)-1})")
                return
            sample = samples[args.sample_idx]

        trace(sample.query, sample, chunk_map, embedder, bm25, reranker_obj,
              cfg, top_k=args.top)


if __name__ == "__main__":
    main()
