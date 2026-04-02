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
from tqdm import tqdm

from config import Config
from dataloader.loader import load_eval_samples
from indexing.chunker import build_chunks
from indexing.embedder import Embedder
from indexing.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever
from reranking.cross_encoder import Reranker
from eval.retrieval_metrics import compute_retrieval_metrics, compute_context_metrics
from eval.answer_metrics import compute_answer_metrics
from eval.reporter import aggregate, save_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")


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
    samples = load_eval_samples(cfg.dataset, cfg.split, cfg.sample)
    logger.info("Evaluating on %d samples.", len(samples))

    # ── 2. Build chunk corpus (with disk cache) ────────────────────────────
    chunk_cache_path = cfg.chunk_cache_dir / "chunks.pkl"
    if chunk_cache_path.exists():
        logger.info("Chunk cache hit: %s", chunk_cache_path)
        with open(chunk_cache_path, "rb") as f:
            unique_chunks = pickle.load(f)
    else:
        logger.info("Chunking documents (strategy=%s) ...", cfg.chunk_strategy)
        all_chunks: list[dict] = []
        for sample in tqdm(samples, desc="Chunking", unit="sample"):
            chunks = build_chunks(sample.documents, cfg)
            all_chunks.extend(chunks)

        # Deduplicate by chunk_id (same doc can appear in multiple samples)
        unique_chunks = list({c["chunk_id"]: c for c in all_chunks}.values())

        cfg.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
        with open(chunk_cache_path, "wb") as f:
            pickle.dump(unique_chunks, f)
        logger.info("Saved chunk cache to %s", chunk_cache_path)

    chunk_map: dict[str, dict] = {c["chunk_id"]: c for c in unique_chunks}
    logger.info("Total unique chunks: %d", len(unique_chunks))

    # ── 3. Build indices ──────────────────────────────────────────────────
    embedder = Embedder(cfg)
    embedder.build_index(unique_chunks)

    bm25 = BM25Index(unique_chunks)

    retriever = HybridRetriever(embedder, bm25, cfg)
    reranker = Reranker(cfg)

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    per_sample_results: list[dict] = []

    for sample in tqdm(samples, desc="Evaluating", unit="query"):
        # Retrieve
        candidates = retriever.retrieve(sample.query, chunk_map)

        # Rerank
        if cfg.reranker != "none":
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

        per_sample_results.append(result)

    # ── 5. Aggregate and save ─────────────────────────────────────────────
    agg = aggregate(per_sample_results)
    logger.info("Aggregated metrics: %s", agg)
    save_run(cfg, per_sample_results, agg)


if __name__ == "__main__":
    main()
