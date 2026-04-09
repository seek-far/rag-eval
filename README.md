# RAG Evaluation System

A from-scratch evaluation framework for comparing RAG pipeline configurations.
No LangChain or LlamaIndex — every component is explicit and inspectable.

## What it does

Runs a full RAG pipeline (chunk → embed → retrieve → rerank) across configurable
datasets and records retrieval + answer quality metrics for every run. Results
accumulate in `results/results.jsonl` so you can compare dozens of experiments
side-by-side with a single command.

---

## Quick start

```bash
# 1. Clone / unzip the project
cd rag-eval

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy example config
cp .env.example .env             # then edit .env as needed

# 5. Run a single experiment
python run.py

# 6. Run the full ablation suite (5 experiments)
chmod +x run_experiments.sh
./run_experiments.sh

# 7. Compare results
python compare_runs.py
```

---

## Requirements

- Python `3.10+`
- Enough disk space for downloaded datasets, embedding caches, and run artifacts
- Optional GPU for faster embedding / reranking (`cuda` or `mps`)

If you are publishing or sharing the repo, keep `.env` private and commit only `.env.example`.

---

## Supported datasets

| Key | Dataset | Answer labels | Notes |
|-----|---------|--------------|-------|
| `nq` | BEIR / Natural Questions | Doc-level relevance | Large open-domain QA |
| `scifact` | BEIR / SciFact | Doc-level relevance | Scientific claim verification |
| `qasper` | QASPER | Extractive spans + free-form | ArXiv NLP papers |
| `quality` | QuALITY | Multiple-choice | Long-document comprehension |

All datasets are downloaded automatically on first run via HuggingFace `datasets`.

---

## All environment variables

### Dataset

| Variable | Default | Options |
|----------|---------|---------|
| `DATASET` | `nq` | `nq` · `scifact` · `qasper` · `quality` |
| `DATASET_SPLIT` | `test` | `train` · `dev` · `test` |
| `DATASET_SAMPLE` | `200` | Any integer; `0` = full dataset |

### Chunking

| Variable | Default | Notes |
|----------|---------|-------|
| `CHUNK_STRATEGY` | `semantic` | `fixed` · `semantic` · `section_then_semantic` |
| `CHUNK_SIZE` | `512` | Token count; used by `fixed` only |
| `CHUNK_OVERLAP` | `64` | Overlap tokens; used by `fixed` only |
| `SEMANTIC_THRESHOLD` | `0.35` | Cosine drop threshold; lower = fewer, larger chunks |
| `CHUNK_MIN_CHARS` | `200` | Merge chunks smaller than this |
| `CHUNK_MAX_CHARS` | `1000` | Hard-split chunks larger than this |

**Chunking strategies:**
- `fixed` — splits every N tokens with overlap. Fast but ignores semantics.
- `semantic` — computes sentence embeddings and cuts at cosine-similarity drops.
- `section_then_semantic` — detects section headers first (best for ArXiv papers),
  then applies semantic splitting within each section.

### Embedding model

| Variable | Default | Recommended alternatives |
|----------|---------|--------------------------|
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | See table below |
| `EMBED_BATCH_SIZE` | `64` | Reduce if OOM |
| `EMBED_DEVICES` | unset | Comma-separated GPUs for multi-device encoding |
| `EMBED_BATCH_MIN` | `8` | Lower bound for auto batch probing |
| `EMBED_BATCH_MAX` | `512` | Upper bound for auto batch probing |
| `EMBED_BATCH_UTILIZATION` | `0.85` | Target GPU memory utilisation when auto-tuning |
| `EMBED_DEVICE` | `cpu` | `cpu` · `cuda` · `mps` |

**Model comparison:**

| Model | Size | Speed | English quality | Multilingual |
|-------|------|-------|----------------|-------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | Fastest | Good | No |
| `intfloat/e5-large-v2` | 335M | Medium | Strong | No |
| `BAAI/bge-large-en-v1.5` | 335M | Medium | Strong | No |
| `BAAI/bge-m3` | 570M | Slow | Strong | Yes |

BGE and E5 models automatically apply the correct query/passage prefixes.

**Multi-GPU encoding:**
- You can distribute embedding and semantic-chunking work across multiple GPUs with `EMBED_DEVICES`.
- Example: `EMBED_DEVICES=cuda:0,cuda:1 python run.py`
- If `EMBED_DEVICES` is set, the project will use those devices for encoding work instead of a single `EMBED_DEVICE`.

**Auto batch-size probing:**
- Set `EMBED_BATCH_SIZE=auto` to let the project probe a safe batch size automatically.
- Tuning is bounded by `EMBED_BATCH_MIN` and `EMBED_BATCH_MAX`.
- `EMBED_BATCH_UTILIZATION` controls how aggressively the probed batch size targets available GPU memory.
- Example: `EMBED_DEVICE=cuda EMBED_BATCH_SIZE=auto python run.py`

### Retrieval

| Variable | Default | Notes |
|----------|---------|-------|
| `RETRIEVAL_MODE` | `hybrid` | `dense` · `sparse` · `hybrid` |
| `RETRIEVE_K` | `20` | Candidates before reranking |
| `RRF_K` | `60` | RRF constant; higher = smoother rank blending |
| `DENSE_WEIGHT` | `0.6` | Must sum to 1.0 with `SPARSE_WEIGHT` |
| `SPARSE_WEIGHT` | `0.4` | Must sum to 1.0 with `DENSE_WEIGHT` |

**When hybrid beats dense-only:** queries with specific terminology (model names,
acronyms, numbers) — BM25 handles exact matches that dense embeddings can miss.

### Reranker

| Variable | Default | Notes |
|----------|---------|-------|
| `RERANKER` | `cross-encoder` | `none` · `cross-encoder` · `rule-top3` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | See options below |
| `RERANK_TOP_K` | `5` | Chunks kept after reranking |

**Reranker model options:**

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast | Good | Default |
| `BAAI/bge-reranker-large` | Slow | Stronger | Better for academic text |

### Evaluation metrics

| Variable | Default |
|----------|---------|
| `EVAL_METRICS` | `mrr,ndcg,hit,recall,f1,bertscore` |
| `EVAL_K_VALUES` | `1,3,5,10` |

**Metrics explained:**
- `mrr@K` — Mean Reciprocal Rank: how early does the first relevant doc appear?
- `ndcg@K` — Normalised Discounted Cumulative Gain: rank-weighted relevance score.
- `hit@K` — Did any relevant doc appear in top K? (binary per query)
- `recall@K` — Fraction of all relevant docs found in top K.
- `f1` — Token-overlap F1 between top-1 chunk text and reference answer.
- `bertscore` — Semantic similarity between predicted and reference answer.

Note: `f1` and `bertscore` only run when a dataset provides reference answers
(QASPER provides extractive/free-form answers; QuALITY provides correct choices).

### Run management

| Variable | Default |
|----------|---------|
| `RUN_NAME` | `default` |
| `RESULTS_DIR` | `./results` |
| `CACHE_DIR` | `./cache` |

Embedding vectors are cached to disk in `CACHE_DIR`. Re-running the same
`EMBED_MODEL` + corpus skips re-encoding and goes straight to retrieval.

---

## GPU / Apple Silicon

**CUDA (NVIDIA):**
```bash
# Replace faiss-cpu with faiss-gpu
pip uninstall faiss-cpu
pip install faiss-gpu

# Install torch with CUDA support (example: CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

EMBED_DEVICE=cuda python run.py
```

**MPS (Apple Silicon):**
```bash
# PyTorch MPS is included in the standard macOS wheel
EMBED_DEVICE=mps python run.py
```

FAISS does not support MPS natively; it will fall back to CPU for the index.
Embedding encoding will use the GPU; search will use CPU — still much faster
than full CPU for large corpora.

---

## Comparing experiments

```bash
# Show all runs
python compare_runs.py

# Filter by dataset
python compare_runs.py --dataset qasper

# Sort by best mrr@10
python compare_runs.py --metric mrr@10

# Show last 5 runs
python compare_runs.py --last 5

# Export to CSV (for Excel / pandas)
python compare_runs.py --csv > results/comparison.csv
```

Example write-up:
- [Natural Questions ablation summary (400 samples)](docs/nq_400_ablation_summary.md)
- [SciFact ablation summary (400 samples)](docs/scifact_400_ablation_summary.md)

---

## Utility scripts

These helper scripts are useful when inspecting runs beyond the aggregate table:

- `python utils/browse_data.py` - inspect dataset samples and fields
- `python utils/trace_query.py` - inspect retrieval / rerank traces for a query
- `python utils/case_analysis.py` - review per-sample successes and failures

---

## Saved Artifacts

Each run still appends one summary row to `results/results.jsonl`, but now it
also saves a per-run artifact directory under:

```text
results/artifacts/<timestamp>_<run_name>/
```

The `results.jsonl` row includes an `artifacts_dir` field pointing to that
directory.

Typical files in each artifact directory:

- `manifest.json` - index of files plus cache paths used by the run
- `summary.json` - run-level summary (dataset, sample count, metrics, chunk count)
- `config_snapshot.json` - config used for the run
- `corpus_summary.json` - corpus/chunk statistics and cache file references
- `sample_inputs.jsonl` - per-sample inputs (`query`, `documents`, labels, answers)
- `per_sample_results.jsonl` - per-sample metrics, predicted answer, and final candidates
- `retrieval_traces.jsonl` - dense/sparse/fused retrieval traces plus rerank inputs/outputs
- `corpus_chunks.jsonl` - all unique chunks used in the run

The existing chunking and embedding caches are unchanged and still live under
`cache/chunks/...` and `cache/embeddings/...`.

---

## Project structure

```
rag-eval/
.env.example                -> copy to .env and configure
config.py                   -> reads all env vars, single source of truth
run.py                      -> main entry point
run_experiments.sh          -> standard ablation suite
compare_runs.py             -> cross-run comparison table
requirements.txt
dataloader/
  schema.py                 -> EvalSample dataclass
  loader.py                 -> NQ / SciFact / QASPER / QuALITY loaders
indexing/
  chunker.py                -> fixed / semantic / section_then_semantic
  embedder.py               -> SentenceTransformer + FAISS + disk cache
  bm25_index.py             -> BM25Okapi sparse index
  encoder_runtime.py        -> device/runtime helpers for encoding
retrieval/
  hybrid.py                 -> dense / sparse / hybrid (RRF)
reranking/
  cross_encoder.py          -> cross-encoder and rule-based rerankers
eval/
  retrieval_metrics.py      -> MRR / NDCG / Hit / Recall
  answer_metrics.py         -> token F1 / BERTScore
  reporter.py               -> saves results.jsonl + prints tables
utils/
  browse_data.py            -> inspect dataset contents
  trace_query.py            -> inspect retrieval traces
  case_analysis.py          -> review qualitative examples
docs/
  *_summary.md              -> shareable experiment write-ups
  figures/                  -> generated figures used in docs
results/                    -> created at runtime; ignored by git
cache/                      -> created at runtime; ignored by git
```

---

## Publishing checklist

Before pushing the repository publicly:

- keep `.env` out of git
- verify model and dataset credentials are not hard-coded anywhere
- avoid committing `results/`, `cache/`, local virtualenvs, or packaging scratch folders
- regenerate figures from the tracked scripts in `docs/` if needed

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Typical ablation results (NQ, n=200)

Running `./run_experiments.sh` produces output like:

```
run_name           dataset  chunk_strategy  embed_model      retrieval  reranker       mrr@10  ndcg@10  hit@10  recall@10
01_baseline        nq       fixed           all-MiniLM-L6    dense      none           0.3124  0.2891   0.6210  0.4830
02_semantic_chunk  nq       semantic        all-MiniLM-L6    dense      none           0.3418  0.3145   0.6580  0.5120
03_bge_dense       nq       semantic        bge-large-en     dense      none           0.4201  0.3890   0.7340  0.5980
04_bge_hybrid      nq       semantic        bge-large-en     hybrid     none           0.4580  0.4210   0.7720  0.6340
05_full_pipeline   nq       semantic        bge-large-en     hybrid     cross-encoder  0.5012  0.4650   0.8100  0.6710
```

Each row isolates the contribution of one change, making it easy to see
which components give the biggest gains on your specific dataset.
