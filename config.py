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


def _parse_device_list(raw: str | None, fallback: str) -> list[str]:
    if raw:
        devices = [item.strip() for item in raw.split(",") if item.strip()]
        if devices:
            return devices
    return [fallback]


def _parse_sample_batches() -> list[tuple[int, int]]:
    """Parse DATASET_SAMPLE, DATASET_SAMPLE_2, ... into [(start, end), ...]."""
    base = int(os.getenv("DATASET_SAMPLE", "200"))
    batches = [(0, base)]
    i = 2
    while True:
        val = os.getenv(f"DATASET_SAMPLE_{i}")
        if val is None:
            break
        size = int(val)
        prev_end = batches[-1][1]
        batches.append((prev_end, prev_end + size))
        i += 1
    return batches


@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: str = os.getenv("DATASET", "nq")
    split: str = os.getenv("DATASET_SPLIT", "test")
    sample: int = int(os.getenv("DATASET_SAMPLE", "200"))
    sample_batches: list = field(default_factory=_parse_sample_batches)

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "semantic")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    semantic_threshold: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.35"))
    chunk_min_chars: int = int(os.getenv("CHUNK_MIN_CHARS", "200"))
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "1000"))

    # ── Embedding ────────────────────────────────────────────────────────────
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
    embed_batch: str = os.getenv("EMBED_BATCH_SIZE", "64")
    embed_device: str = os.getenv("EMBED_DEVICE", "cpu")
    embed_devices_raw: str = os.getenv("EMBED_DEVICES", "")
    embed_batch_min: int = int(os.getenv("EMBED_BATCH_MIN", "8"))
    embed_batch_max: int = int(os.getenv("EMBED_BATCH_MAX", "512"))
    embed_batch_utilization: float = float(
        os.getenv("EMBED_BATCH_UTILIZATION", "0.85")
    )

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
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))
    llm_rerank_max_chars: int = int(os.getenv("LLM_RERANK_MAX_CHARS", "1500"))

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
        self.embed_devices = _parse_device_list(self.embed_devices_raw, self.embed_device)
        raw_batch = str(self.embed_batch).strip()
        self.embed_batch_is_auto = raw_batch.lower() == "auto"
        self.embed_batch = 32 if self.embed_batch_is_auto else int(raw_batch)
        self.embed_batch_min = max(1, self.embed_batch_min)
        self.embed_batch_max = max(self.embed_batch_min, self.embed_batch_max)
        self.embed_batch_utilization = min(max(self.embed_batch_utilization, 0.1), 0.98)

    @property
    def _chunk_params_tag(self) -> str:
        """Hash tag encoding chunk strategy + parameters."""
        if self.chunk_strategy == "fixed":
            params = f"sz{self.chunk_size}_ov{self.chunk_overlap}"
        elif self.chunk_strategy == "sentence":
            params = f"min{self.chunk_min_chars}_max{self.chunk_max_chars}"
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
        return f"{self.chunk_strategy}_{digest}"

    @property
    def chunk_cache_dir(self) -> Path:
        """Cache dir for chunking results. Batch files live inside."""
        dataset_part = f"{self.dataset}_{self.split}"
        return Path(self.cache_dir) / "chunks" / dataset_part / self._chunk_params_tag

    @property
    def embed_cache_dir(self) -> Path:
        """Cache dir for embedding vectors (chunk_id → vector dict)."""
        dataset_part = f"{self.dataset}_{self.split}"
        model_safe = self.embed_model.replace("/", "_")
        return Path(self.cache_dir) / "embeddings" / dataset_part / self._chunk_params_tag / model_safe

    @property
    def total_samples(self) -> int:
        return self.sample_batches[-1][1]

    def summary(self) -> str:
        batch_desc = " + ".join(f"{e-s}" for s, e in self.sample_batches)
        device_desc = ",".join(self.embed_devices)
        batch_desc_embed = "auto" if self.embed_batch_is_auto else str(self.embed_batch)
        return (
            f"  dataset        : {self.dataset} / {self.split} "
            f"(samples={batch_desc}, total={self.total_samples})\n"
            f"  chunk_strategy : {self.chunk_strategy} "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap}, "
            f"threshold={self.semantic_threshold})\n"
            f"  embed_model    : {self.embed_model} "
            f"(device={self.embed_device}, devices={device_desc}, batch={batch_desc_embed})\n"
            f"  retrieval      : {self.retrieval_mode} "
            f"(k={self.retrieve_k}, rrf_k={self.rrf_k}, "
            f"dense={self.dense_weight}, sparse={self.sparse_weight})\n"
            f"  reranker       : {self.reranker} "
            f"(model={self.reranker_model}, top_k={self.rerank_top_k})\n"
            f"  llm_reranker   : model={self.llm_model or '-'} "
            f"(base_url={self.llm_base_url or '-'}, max_chars={self.llm_rerank_max_chars})\n"
            f"  metrics        : {', '.join(self.metrics)} "
            f"@ k={self.k_values}\n"
            f"  run_name       : {self.run_name}\n"
            f"  results_dir    : {self.results_dir}\n"
        )
