#!/usr/bin/env bash
# run_experiments.sh
#
# Runs a standard ablation suite: baseline → +semantic → +BGE → +hybrid → full
# Each run appends to results/results.jsonl automatically.
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh            # default dataset=nq, sample=200
#   DATASET=scifact ./run_experiments.sh
#   DATASET=qasper DATASET_SAMPLE=100 ./run_experiments.sh

set -e

DATASET=${DATASET:-nq}
SPLIT=${DATASET_SPLIT:-test}
SAMPLE=${DATASET_SAMPLE:-200}
DEVICE=${EMBED_DEVICE:-cpu}

echo "=========================================="
echo "  RAG Ablation Suite"
echo "  dataset=$DATASET  split=$SPLIT  sample=$SAMPLE  device=$DEVICE"
echo "=========================================="

# ── 1. Baseline: fixed chunks + MiniLM + dense ──────────────────────────
DATASET=$DATASET DATASET_SPLIT=$SPLIT DATASET_SAMPLE=$SAMPLE \
EMBED_DEVICE=$DEVICE \
RUN_NAME="01_baseline" \
CHUNK_STRATEGY=fixed \
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
RETRIEVAL_MODE=dense \
RERANKER=none \
python run.py

# ── 2. + Semantic chunking ───────────────────────────────────────────────
DATASET=$DATASET DATASET_SPLIT=$SPLIT DATASET_SAMPLE=$SAMPLE \
EMBED_DEVICE=$DEVICE \
RUN_NAME="02_semantic_chunk" \
CHUNK_STRATEGY=semantic \
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
RETRIEVAL_MODE=dense \
RERANKER=none \
python run.py

# ── 3. + Better embedding (BGE) ──────────────────────────────────────────
DATASET=$DATASET DATASET_SPLIT=$SPLIT DATASET_SAMPLE=$SAMPLE \
EMBED_DEVICE=$DEVICE \
RUN_NAME="03_bge_dense" \
CHUNK_STRATEGY=semantic \
EMBED_MODEL=BAAI/bge-large-en-v1.5 \
RETRIEVAL_MODE=dense \
RERANKER=none \
python run.py

# ── 4. + Hybrid retrieval (RRF) ──────────────────────────────────────────
DATASET=$DATASET DATASET_SPLIT=$SPLIT DATASET_SAMPLE=$SAMPLE \
EMBED_DEVICE=$DEVICE \
RUN_NAME="04_bge_hybrid" \
CHUNK_STRATEGY=semantic \
EMBED_MODEL=BAAI/bge-large-en-v1.5 \
RETRIEVAL_MODE=hybrid \
RERANKER=none \
python run.py

# ── 5. Full pipeline: + cross-encoder reranker ───────────────────────────
DATASET=$DATASET DATASET_SPLIT=$SPLIT DATASET_SAMPLE=$SAMPLE \
EMBED_DEVICE=$DEVICE \
RUN_NAME="05_full_pipeline" \
CHUNK_STRATEGY=semantic \
EMBED_MODEL=BAAI/bge-large-en-v1.5 \
RETRIEVAL_MODE=hybrid \
RERANKER=cross-encoder \
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
RERANK_TOP_K=5 \
python run.py

echo ""
echo "All runs complete. Comparison table:"
python compare_runs.py --dataset "$DATASET"
