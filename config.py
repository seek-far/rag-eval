"""
config.py — Reads all settings from environment variables.
Copy .env.example to .env and adjust, or export variables directly.
"""
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _list_str(key: str, default: str) -> list[str]:
    return os.getenv(key, default).split(",")


def _list_int(key: str, default: str) -> list[int]:
    return [int(x) for x in os.getenv(key, default).split(",")]


@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: str = os.getenv("DATASET", "nq")
    split: str = os.getenv("DATASET_SPLIT", "test")
    sample: int = int(os.getenv("DATASET_SAMPLE", "200"))

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "semantic")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    semantic_threshold: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.35"))
    chunk_min_chars: int = int(os.getenv("CHUNK_MIN_CHARS", "200"))
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "1000"))

    # ── Embedding ────────────────────────────────────────────────────────────
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    embed_batch: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    embed_device: str = os.getenv("EMBED_DEVICE", "cpu")

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")
    retrieve_k: int = int(os.getenv("RETRIEVE_K", "20"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "0.6"))
    sparse_weight: float = float(os.getenv("SPARSE_WEIGHT", "0.4"))

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker: str = os.getenv("RERANKER", "cross-encoder")
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))

    # ── Eval ─────────────────────────────────────────────────────────────────
    metrics: list = field(
        default_factory=lambda: _list_str(
            "EVAL_METRICS", "mrr,ndcg,hit,recall,f1,bertscore"
        )
    )
    k_values: list = field(
        default_factory=lambda: _list_int("EVAL_K_VALUES", "1,3,5,10")
    )

    # ── Run management ───────────────────────────────────────────────────────
    run_name: str = os.getenv("RUN_NAME", "default")
    results_dir: str = os.getenv("RESULTS_DIR", "./results")
    cache_dir: str = os.getenv("CACHE_DIR", "./cache")

    def __post_init__(self):
        # Re-read list fields after init (dataclass field() + env interaction)
        if isinstance(self.metrics, list) and self.metrics == []:
            self.metrics = _list_str("EVAL_METRICS", "mrr,ndcg,hit,recall,f1,bertscore")
        if isinstance(self.k_values, list) and self.k_values == []:
            self.k_values = _list_int("EVAL_K_VALUES", "1,3,5,10")

    @property
    def chunk_cache_dir(self) -> Path:
        """Cache dir for chunking results, determined by dataset + chunk params."""
        dataset_part = f"{self.dataset}_{self.split}_n{self.sample}"
        if self.chunk_strategy == "fixed":
            params = f"sz{self.chunk_size}_ov{self.chunk_overlap}"
        else:
            # semantic / section_then_semantic — includes embed_model
            # because semantic chunking uses it for sentence similarity
            model_short = self.embed_model.replace("/", "_")
            params = (
                f"th{self.semantic_threshold}_min{self.chunk_min_chars}"
                f"_max{self.chunk_max_chars}_{model_short}"
            )
        tag = f"{self.chunk_strategy}_{params}"
        digest = hashlib.md5(tag.encode()).hexdigest()[:8]
        return Path(self.cache_dir) / "chunks" / dataset_part / f"{self.chunk_strategy}_{digest}"

    @property
    def embed_cache_dir(self) -> Path:
        """Cache dir for embedding vectors, nested under the chunk cache."""
        model_safe = self.embed_model.replace("/", "_")
        return Path(self.cache_dir) / "embeddings" / self.chunk_cache_dir.relative_to(
            Path(self.cache_dir) / "chunks"
        ) / model_safe

    def summary(self) -> str:
        return (
            f"  dataset        : {self.dataset} / {self.split} "
            f"(sample={self.sample or 'all'})\n"
            f"  chunk_strategy : {self.chunk_strategy} "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap}, "
            f"threshold={self.semantic_threshold})\n"
            f"  embed_model    : {self.embed_model} "
            f"(device={self.embed_device}, batch={self.embed_batch})\n"
            f"  retrieval      : {self.retrieval_mode} "
            f"(k={self.retrieve_k}, rrf_k={self.rrf_k}, "
            f"dense={self.dense_weight}, sparse={self.sparse_weight})\n"
            f"  reranker       : {self.reranker} "
            f"(model={self.reranker_model}, top_k={self.rerank_top_k})\n"
            f"  metrics        : {', '.join(self.metrics)} "
            f"@ k={self.k_values}\n"
            f"  run_name       : {self.run_name}\n"
            f"  results_dir    : {self.results_dir}\n"
        )
