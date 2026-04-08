"""
run.py — Main entry point for a single evaluation run.

Usage:
    python run.py                          # uses .env or environment variables
    RUN_NAME=exp1 DATASET=nq python run.py
    python run.py --help                   # show current config and exit
"""
from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

from config import Config
from dataloader.loader import load_eval_samples
from indexing.chunker import build_chunks, create_chunker
from indexing.embedder import Embedder
from indexing.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever, _rrf
from reranking.cross_encoder import Reranker
from eval.retrieval_metrics import compute_retrieval_metrics, compute_context_metrics
from eval.answer_metrics import compute_answer_metrics
from eval.reporter import aggregate, save_run

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")


def _snapshot_chunk(chunk: dict, *, rank: int | None = None, relevant_doc_ids=None) -> dict:
    relevant_set = set(relevant_doc_ids or [])
    row = {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "section_type": chunk.get("section_type", ""),
        "text": chunk["text"],
        "doc_text": chunk.get("doc_text", ""),
        "section_text": chunk.get("section_text", ""),
        "retrieval_score": float(chunk["retrieval_score"]) if "retrieval_score" in chunk else None,
        "rerank_score": float(chunk["rerank_score"]) if "rerank_score" in chunk else None,
        "is_relevant_doc": chunk["doc_id"] in relevant_set,
    }
    if rank is not None:
        row["rank"] = rank
    return row


def _sample_input_row(sample) -> dict:
    return {
        "sample_id": sample.id,
        "query": sample.query,
        "documents": sample.documents,
        "document_count": len(sample.documents),
        "relevant_doc_ids": sample.relevant_doc_ids,
        "reference_answer": sample.reference_answer,
        "answer_spans": sample.answer_spans,
        "choices": sample.choices,
        "correct_choice": sample.correct_choice,
    }


def _stage_trace(
    ranked: list[tuple[str, float]],
    chunk_map: dict[str, dict],
    limit: int,
) -> list[dict]:
    rows = []
    for rank, (chunk_id, score) in enumerate(ranked[:limit], start=1):
        if chunk_id not in chunk_map:
            continue
        chunk = chunk_map[chunk_id]
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "doc_id": chunk["doc_id"],
                "score": float(score),
                "section_type": chunk.get("section_type", ""),
                "text": chunk["text"],
            }
        )
    return rows


def _build_retrieval_trace(retriever, query: str, chunk_map: dict[str, dict]) -> tuple[list[dict], dict]:
    dense_results: list[tuple[str, float]] = []
    sparse_results: list[tuple[str, float]] = []
    if retriever.mode == "dense":
        fused_results = retriever._dense(query)
        dense_results = fused_results
    elif retriever.mode == "sparse":
        fused_results = retriever._sparse(query)
        sparse_results = fused_results
    else:
        dense_results = retriever._dense(query)
        sparse_results = retriever._sparse(query)
        fused_results = _rrf(
            dense_results,
            sparse_results,
            k=retriever.rrf_k,
            w_dense=retriever.dense_w,
            w_sparse=retriever.sparse_w,
        )

    candidates = []
    for chunk_id, score in fused_results[: retriever.k]:
        if chunk_id not in chunk_map:
            continue
        chunk = dict(chunk_map[chunk_id])
        chunk["retrieval_score"] = float(score)
        candidates.append(chunk)

    trace = {
        "mode": retriever.mode,
        "dense": _stage_trace(dense_results, chunk_map, retriever.k),
        "sparse": _stage_trace(sparse_results, chunk_map, retriever.k),
        "fused": _stage_trace(fused_results, chunk_map, retriever.k),
    }
    return candidates, trace


def _corpus_summary(samples, unique_chunks: list[dict], cfg) -> dict:
    unique_doc_ids = sorted({doc["id"] for sample in samples for doc in sample.documents})
    section_counts: dict[str, int] = {}
    chunk_lengths: list[int] = []
    for chunk in unique_chunks:
        section = chunk.get("section_type", "body")
        section_counts[section] = section_counts.get(section, 0) + 1
        chunk_lengths.append(len(chunk["text"]))

    chunk_cache_files = sorted(str(p) for p in Path(cfg.chunk_cache_dir).glob("*.pkl"))
    embed_cache_files = sorted(str(p) for p in Path(cfg.embed_cache_dir).glob("*"))
    return {
        "sample_count": len(samples),
        "unique_document_count": len(unique_doc_ids),
        "unique_chunk_count": len(unique_chunks),
        "section_counts": section_counts,
        "avg_chunk_chars": (sum(chunk_lengths) / len(chunk_lengths)) if chunk_lengths else 0.0,
        "max_chunk_chars": max(chunk_lengths) if chunk_lengths else 0,
        "min_chunk_chars": min(chunk_lengths) if chunk_lengths else 0,
        "chunk_cache_dir": str(cfg.chunk_cache_dir),
        "chunk_cache_files": chunk_cache_files,
        "embed_cache_dir": str(cfg.embed_cache_dir),
        "embed_cache_files": embed_cache_files,
        "sample_batches": list(cfg.sample_batches),
    }


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        cfg = Config()
        print("Current configuration:\n")
        print(cfg.summary())
        print("Set environment variables (or edit .env) to change any value.")
        return

    cfg = Config()
    print(f"\n{'='*60}")
    print(f"  RAG Evaluation  —  run: {cfg.run_name}")
    print(f"{'='*60}")
    print(cfg.summary())

    # ── 1. Load dataset ───────────────────────────────────────────────────
    all_samples = load_eval_samples(cfg.dataset, cfg.split, cfg.total_samples)
    samples = all_samples[: cfg.total_samples]
    logger.info("Evaluating on %d samples (%d batches).",
                len(samples), len(cfg.sample_batches))

    # ── 2. Build chunk corpus (per-batch caching) ─────────────────────────
    cfg.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_map: dict[str, dict] = {}
    chunker = None  # lazy init, only if needed

    for batch_start, batch_end in cfg.sample_batches:
        cache_path = cfg.chunk_cache_dir / f"batch_{batch_start}_{batch_end}.pkl"
        if cache_path.exists():
            logger.info("Chunk cache hit: %s", cache_path.name)
            with open(cache_path, "rb") as f:
                batch_chunks = pickle.load(f)
        else:
            batch_samples = samples[batch_start:batch_end]
            logger.info(
                "Chunking batch [%d:%d] (%d samples, strategy=%s) ...",
                batch_start, batch_end, len(batch_samples), cfg.chunk_strategy,
            )
            if chunker is None:
                chunker = create_chunker(cfg)
            all_chunks: list[dict] = []
            for sample in tqdm(batch_samples, desc=f"Chunking [{batch_start}:{batch_end}]",
                               unit="sample"):
                chunks = build_chunks(sample.documents, cfg, chunker=chunker)
                all_chunks.extend(chunks)
            batch_chunks = list({c["chunk_id"]: c for c in all_chunks}.values())
            if batch_chunks:
                with open(cache_path, "wb") as f:
                    pickle.dump(batch_chunks, f)
                logger.info("Saved chunk cache: %s (%d chunks)", cache_path.name, len(batch_chunks))
            else:
                logger.warning("Batch [%d:%d] produced 0 chunks, skipping cache.", batch_start, batch_end)

        for c in batch_chunks:
            chunk_map.setdefault(c["chunk_id"], c)

    unique_chunks = list(chunk_map.values())
    logger.info("Total unique chunks: %d", len(unique_chunks))

    # ── 3. Build indices ──────────────────────────────────────────────────
    embedder = Embedder(cfg)
    embedder.build_index(unique_chunks)

    bm25 = BM25Index(unique_chunks)

    retriever = HybridRetriever(embedder, bm25, cfg)
    reranker = Reranker(cfg) if cfg.reranker != "none" else None

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    per_sample_results: list[dict] = []
    sample_inputs: list[dict] = [_sample_input_row(sample) for sample in samples]
    retrieval_traces: list[dict] = []

    for sample in tqdm(samples, desc="Evaluating", unit="query"):
        # Retrieve
        candidates, retrieval_trace = _build_retrieval_trace(retriever, sample.query, chunk_map)
        retrieved_pre_rerank = [
            _snapshot_chunk(c, rank=rank, relevant_doc_ids=sample.relevant_doc_ids)
            for rank, c in enumerate(candidates, start=1)
        ]

        # Rerank
        if reranker is not None:
            candidates = reranker.rerank(sample.query, candidates)
            candidates = candidates[: cfg.rerank_top_k]

        # Deduplicate doc_ids while preserving rank order
        seen_doc_ids: set[str] = set()
        retrieved_ids: list[str] = []
        for c in candidates:
            if c["doc_id"] not in seen_doc_ids:
                seen_doc_ids.add(c["doc_id"])
                retrieved_ids.append(c["doc_id"])

        result: dict[str, float] = {}
        predicted = ""

        # Retrieval metrics (need relevant_doc_ids)
        if sample.relevant_doc_ids:
            result.update(
                compute_retrieval_metrics(
                    retrieved_ids,
                    sample.relevant_doc_ids,
                    cfg.k_values,
                    cfg.metrics,
                )
            )

        # Answer-in-context (need answer_spans)
        if sample.answer_spans and candidates:
            chunk_texts = [c["text"] for c in candidates]
            result.update(
                compute_context_metrics(chunk_texts, sample.answer_spans, cfg.k_values)
            )

        # Answer-quality metrics (need reference_answer)
        if sample.reference_answer and candidates:
            # Use the top-1 retrieved chunk text as the predicted answer
            predicted = candidates[0]["text"]
            result.update(
                compute_answer_metrics(
                    predicted,
                    sample.reference_answer,
                    cfg.metrics,
                )
            )

        final_candidates = [
            _snapshot_chunk(c, rank=rank, relevant_doc_ids=sample.relevant_doc_ids)
            for rank, c in enumerate(candidates, start=1)
        ]
        sample_result = {
            "sample_id": sample.id,
            "query": sample.query,
            "document_count": len(sample.documents),
            "relevant_doc_ids": sample.relevant_doc_ids,
            "retrieved_doc_ids": retrieved_ids,
            "predicted_answer": predicted,
            "reference_answer": sample.reference_answer,
            "answer_spans": sample.answer_spans,
            "metrics": result,
            "pre_rerank_candidates": retrieved_pre_rerank,
            "final_candidates": final_candidates,
        }
        per_sample_results.append(sample_result)
        retrieval_traces.append(
            {
                "sample_id": sample.id,
                "query": sample.query,
                "retrieval_mode": cfg.retrieval_mode,
                "retrieval_trace": retrieval_trace,
                "pre_rerank_candidates": retrieved_pre_rerank,
                "final_candidates": final_candidates,
                "retrieved_doc_ids": retrieved_ids,
            }
        )

    # ── 5. Aggregate and save ─────────────────────────────────────────────
    metric_rows = [row["metrics"] for row in per_sample_results]
    agg = aggregate(metric_rows)
    logger.info("Aggregated metrics: %s", agg)
    corpus_rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "section_type": chunk.get("section_type", ""),
            "text": chunk["text"],
            "doc_text": chunk.get("doc_text", ""),
            "section_text": chunk.get("section_text", ""),
        }
        for chunk in unique_chunks
    ]
    artifacts = {
        "summary": {
            "run_name": cfg.run_name,
            "dataset": cfg.dataset,
            "split": cfg.split,
            "n_samples": len(samples),
            "n_unique_chunks": len(unique_chunks),
            "metrics": agg,
        },
        "config_snapshot": {
            "dataset": cfg.dataset,
            "split": cfg.split,
            "sample": cfg.sample,
            "sample_batches": list(cfg.sample_batches),
            "chunk_strategy": cfg.chunk_strategy,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "semantic_threshold": cfg.semantic_threshold,
            "chunk_min_chars": cfg.chunk_min_chars,
            "chunk_max_chars": cfg.chunk_max_chars,
            "embed_model": cfg.embed_model,
            "embed_device": cfg.embed_device,
            "embed_devices": cfg.embed_devices,
            "embed_batch": cfg.embed_batch,
            "embed_batch_is_auto": cfg.embed_batch_is_auto,
            "retrieve_k": cfg.retrieve_k,
            "retrieval_mode": cfg.retrieval_mode,
            "rrf_k": cfg.rrf_k,
            "dense_weight": cfg.dense_weight,
            "sparse_weight": cfg.sparse_weight,
            "reranker": cfg.reranker,
            "reranker_model": cfg.reranker_model,
            "rerank_top_k": cfg.rerank_top_k,
            "metrics": cfg.metrics,
            "k_values": cfg.k_values,
            "results_dir": cfg.results_dir,
            "cache_dir": cfg.cache_dir,
            "chunk_cache_dir": str(cfg.chunk_cache_dir),
            "embed_cache_dir": str(cfg.embed_cache_dir),
        },
        "corpus_summary": _corpus_summary(samples, unique_chunks, cfg),
        "sample_inputs": sample_inputs,
        "per_sample_results": per_sample_results,
        "retrieval_traces": retrieval_traces,
        "corpus_chunks": corpus_rows,
    }
    save_run(cfg, metric_rows, agg, artifacts=artifacts)


if __name__ == "__main__":
    main()
