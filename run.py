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
from indexing.chunker import build_chunks, create_chunker
from indexing.embedder import Embedder
from indexing.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever
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

    for sample in tqdm(samples, desc="Evaluating", unit="query"):
        # Retrieve
        candidates = retriever.retrieve(sample.query, chunk_map)

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
