"""
chunker.py — Chunking strategies controlled via Config.

  fixed               : split by token count with overlap
  sentence            : split at sentence boundaries, group by size (no embedding)
  semantic            : split at cosine-similarity drop points
  section_then_semantic : detect section headers first, then semantic-split
                         each section (best for ArXiv / structured docs)

Every chunk carries full metadata so context_builder can do parent-chunk
retrieval without an extra DB lookup.
"""
from __future__ import annotations

import re
import logging
import numpy as np

from indexing.encoder_runtime import SentenceEncoderRuntime

logger = logging.getLogger(__name__)

# ── Section header patterns (covers ArXiv / academic papers) ──────────────
_SECTION_RE = re.compile(
    r"^(?:"
    r"#{1,3}\s+.+"                        # Markdown ## / ###
    r"|(?:\d+\.?)+\s+[A-Z][A-Za-z\s]+"   # 1. / 1.1 Introduction
    r"|[A-Z][A-Z\s]{3,40}$"              # ALL-CAPS HEADER
    r"|(?:Abstract|Introduction|Background|Related Work|"
    r"Methodology|Methods?|Experiments?|Results?|"
    r"Discussion|Conclusion|References|Appendix)"
    r"[\s\n:]"
    r")",
    re.MULTILINE,
)

# Sections we skip during indexing (noisy, no QA value)
_SKIP_SECTIONS = {"references", "appendix"}


# ─────────────────────────── Public entry point ───────────────────────────

def create_chunker(cfg):
    """Create a chunker instance once; reuse across samples to avoid reloading models."""
    strategy = cfg.chunk_strategy
    if strategy == "fixed":
        return _FixedChunker(cfg.chunk_size, cfg.chunk_overlap)
    elif strategy == "sentence":
        return _SentenceChunker(cfg.chunk_min_chars, cfg.chunk_max_chars)
    elif strategy == "semantic":
        return _SemanticChunker(
            cfg.embed_model,
            cfg.embed_device,
            cfg.embed_devices,
            cfg.embed_batch,
            cfg.embed_batch_is_auto,
            cfg.embed_batch_min,
            cfg.embed_batch_max,
            cfg.embed_batch_utilization,
            cfg.semantic_threshold,
            cfg.chunk_min_chars,
            cfg.chunk_max_chars,
        )
    elif strategy == "section_then_semantic":
        return _SectionSemanticChunker(
            cfg.embed_model,
            cfg.embed_device,
            cfg.embed_devices,
            cfg.embed_batch,
            cfg.embed_batch_is_auto,
            cfg.embed_batch_min,
            cfg.embed_batch_max,
            cfg.embed_batch_utilization,
            cfg.semantic_threshold,
            cfg.chunk_min_chars,
            cfg.chunk_max_chars,
        )
    else:
        raise ValueError(f"Unknown chunk_strategy: {strategy!r}")


def build_chunks(
    documents: list[dict],  # [{"id": str, "text": str}, ...]
    cfg,
    chunker=None,
) -> list[dict]:
    """
    Returns a flat list of chunk dicts.  Each chunk has:
      chunk_id        unique string
      doc_id          source document id
      doc_text        full source document text (for parent-chunk retrieval)
      section_type    normalised section name (semantic/section mode)
      section_text    full section text (for parent-chunk retrieval)
      text            the chunk text used for embedding & BM25
    """
    if chunker is None:
        chunker = create_chunker(cfg)

    if hasattr(chunker, "batch_chunk"):
        all_chunks = chunker.batch_chunk(documents)
        logger.debug(
            "Built %d chunks from %d documents (strategy=%s)",
            len(all_chunks),
            len(documents),
            cfg.chunk_strategy,
        )
        return all_chunks

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc["text"], doc["id"])
        all_chunks.extend(chunks)

    logger.debug(
        "Built %d chunks from %d documents (strategy=%s)",
        len(all_chunks),
        len(documents),
        cfg.chunk_strategy,
    )
    return all_chunks


# ─────────────────────────── Fixed chunker ────────────────────────────────

class _FixedChunker:
    def __init__(self, size: int, overlap: int):
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        words = text.split()
        step = max(1, self.size - self.overlap)
        chunks = []
        for i, start in enumerate(range(0, len(words), step)):
            chunk_text = " ".join(words[start : start + self.size])
            if len(chunk_text) < 20:
                continue
            chunks.append(
                _make_chunk(
                    doc_id=doc_id,
                    doc_text=text,
                    section_type="body",
                    section_text=text,
                    chunk_idx=i,
                    text=chunk_text,
                )
            )
        return chunks


# ─────────────────────────── Sentence chunker ────────────────────────────

class _SentenceChunker:
    """Group sentences by size only — no embedding model needed."""

    def __init__(self, min_chars: int, max_chars: int):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        sentences = _split_sentences_robust(text)
        if not sentences:
            return [_make_chunk(doc_id, text, "body", text, 0, text)]

        groups: list[list[str]] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            # Start a new group if adding this sentence would exceed max_chars
            if current and current_len + sent_len + 1 > self.max_chars:
                groups.append(current)
                current, current_len = [], 0
            current.append(sent)
            current_len += sent_len + 1  # +1 for joining space

        if current:
            # Merge trailing short group into previous
            if len(" ".join(current)) < self.min_chars and groups:
                groups[-1].extend(current)
            else:
                groups.append(current)

        return [
            _make_chunk(doc_id, text, "body", text, i, " ".join(g))
            for i, g in enumerate(groups)
        ]


# ─────────────────────────── Semantic chunker ─────────────────────────────

class _SemanticChunker:
    def __init__(
        self,
        model_name: str,
        device: str,
        devices: list[str],
        batch_size: int,
        auto_batch: bool,
        batch_min: int,
        batch_max: int,
        batch_utilization: float,
        threshold: float,
        min_chars: int,
        max_chars: int,
    ):
        self.runtime = SentenceEncoderRuntime(
            model_name=model_name,
            primary_device=devices[0] if devices else device,
            devices=devices or [device],
            batch_size=batch_size,
            auto_batch=auto_batch,
            batch_min=batch_min,
            batch_max=batch_max,
            batch_utilization=batch_utilization,
            stage_name="semantic chunking",
        )
        self.threshold = threshold
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        sentences = _split_sentences(text)
        if len(sentences) <= 2:
            return [_make_chunk(doc_id, text, "body", text, 0, text)]

        embeddings = self.runtime.encode(
            sentences,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        groups = self._group(sentences, embeddings)
        return [
            _make_chunk(doc_id, text, "body", text, i, " ".join(g))
            for i, g in enumerate(groups)
        ]

    def batch_chunk(self, documents: list[dict]) -> list[dict]:
        prepared = []
        batched_sentences: list[str] = []

        for doc in documents:
            text = doc["text"]
            sentences = _split_sentences(text)
            prepared.append(
                {
                    "doc_id": doc["id"],
                    "doc_text": text,
                    "sentences": sentences,
                }
            )
            if len(sentences) > 2:
                batched_sentences.extend(sentences)

        batched_embeddings = self._encode_sentences(batched_sentences)
        cursor = 0
        all_chunks: list[dict] = []

        for item in prepared:
            doc_id = item["doc_id"]
            doc_text = item["doc_text"]
            sentences = item["sentences"]

            if len(sentences) <= 2:
                all_chunks.append(_make_chunk(doc_id, doc_text, "body", doc_text, 0, doc_text))
                continue

            next_cursor = cursor + len(sentences)
            doc_embeddings = batched_embeddings[cursor:next_cursor]
            cursor = next_cursor
            groups = self._group(sentences, doc_embeddings)
            all_chunks.extend(
                _make_chunk(doc_id, doc_text, "body", doc_text, i, " ".join(g))
                for i, g in enumerate(groups)
            )

        return all_chunks

    def __del__(self) -> None:
        try:
            self.runtime.close()
        except Exception:
            pass

    def _encode_sentences(self, sentences: list[str]) -> np.ndarray:
        if not sentences:
            return np.empty((0, 0), dtype="float32")
        return self.runtime.encode(
            sentences,
            normalize_embeddings=False,
            show_progress_bar=False,
        )

    def _group(
        self, sentences: list[str], embeddings: np.ndarray
    ) -> list[list[str]]:
        breakpoints: set[int] = set()
        for i in range(len(embeddings) - 1):
            sim = _cosine(embeddings[i], embeddings[i + 1])
            if sim < self.threshold:
                breakpoints.add(i)

        groups: list[list[str]] = []
        current: list[str] = []
        for i, sent in enumerate(sentences):
            current.append(sent)
            if i in breakpoints:
                merged = " ".join(current)
                if len(merged) < self.min_chars and groups:
                    groups[-1].extend(current)
                else:
                    groups.append(current)
                current = []
        if current:
            groups.append(current)

        # Hard-split groups that exceed max_chars
        final: list[list[str]] = []
        for g in groups:
            if len(" ".join(g)) > self.max_chars:
                final.extend(_hard_split(g, self.max_chars))
            else:
                final.append(g)
        return final


# ─────────────── Section-then-semantic chunker ────────────────────────────

class _SectionSemanticChunker:
    """
    Stage 1: split text by section headers → list of (section_type, text)
    Stage 2: apply SemanticChunker within each section
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        devices: list[str],
        batch_size: int,
        auto_batch: bool,
        batch_min: int,
        batch_max: int,
        batch_utilization: float,
        threshold: float,
        min_chars: int,
        max_chars: int,
    ):
        self._sem = _SemanticChunker(
            model_name,
            device,
            devices,
            batch_size,
            auto_batch,
            batch_min,
            batch_max,
            batch_utilization,
            threshold,
            min_chars,
            max_chars,
        )

    def chunk(self, text: str, doc_id: str) -> list[dict]:
        sections = _split_sections(text)
        all_chunks: list[dict] = []
        chunk_idx = 0
        for sec_type, sec_text in sections:
            if sec_type in _SKIP_SECTIONS:
                continue
            sentences = _split_sentences(sec_text)
            if not sentences:
                continue
            if len(sentences) <= 2:
                all_chunks.append(
                    _make_chunk(doc_id, text, sec_type, sec_text, chunk_idx, sec_text)
                )
                chunk_idx += 1
                continue
            embeddings = self._sem.runtime.encode(
                sentences,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            groups = self._sem._group(sentences, embeddings)
            for g in groups:
                all_chunks.append(
                    _make_chunk(
                        doc_id, text, sec_type, sec_text, chunk_idx, " ".join(g)
                    )
                )
                chunk_idx += 1
        return all_chunks

    def batch_chunk(self, documents: list[dict]) -> list[dict]:
        prepared = []
        batched_sentences: list[str] = []

        for doc in documents:
            doc_id = doc["id"]
            doc_text = doc["text"]
            sections = []
            for sec_type, sec_text in _split_sections(doc_text):
                if sec_type in _SKIP_SECTIONS:
                    continue
                sentences = _split_sentences(sec_text)
                sections.append(
                    {
                        "section_type": sec_type,
                        "section_text": sec_text,
                        "sentences": sentences,
                    }
                )
                if len(sentences) > 2:
                    batched_sentences.extend(sentences)
            prepared.append(
                {
                    "doc_id": doc_id,
                    "doc_text": doc_text,
                    "sections": sections,
                }
            )

        batched_embeddings = self._sem._encode_sentences(batched_sentences)
        cursor = 0
        all_chunks: list[dict] = []

        for item in prepared:
            chunk_idx = 0
            for section in item["sections"]:
                sec_type = section["section_type"]
                sec_text = section["section_text"]
                sentences = section["sentences"]

                if not sentences:
                    continue
                if len(sentences) <= 2:
                    all_chunks.append(
                        _make_chunk(
                            item["doc_id"],
                            item["doc_text"],
                            sec_type,
                            sec_text,
                            chunk_idx,
                            sec_text,
                        )
                    )
                    chunk_idx += 1
                    continue

                next_cursor = cursor + len(sentences)
                section_embeddings = batched_embeddings[cursor:next_cursor]
                cursor = next_cursor
                groups = self._sem._group(sentences, section_embeddings)
                for group in groups:
                    all_chunks.append(
                        _make_chunk(
                            item["doc_id"],
                            item["doc_text"],
                            sec_type,
                            sec_text,
                            chunk_idx,
                            " ".join(group),
                        )
                    )
                    chunk_idx += 1

        return all_chunks

    def __del__(self) -> None:
        try:
            self._sem.runtime.close()
        except Exception:
            pass


# ─────────────────────────── Helpers ──────────────────────────────────────

def _make_chunk(
    doc_id: str,
    doc_text: str,
    section_type: str,
    section_text: str,
    chunk_idx: int,
    text: str,
) -> dict:
    return {
        "chunk_id": f"{doc_id}__c{chunk_idx}",
        "doc_id": doc_id,
        "doc_text": doc_text,          # full doc — for parent-chunk retrieval
        "section_type": section_type,
        "section_text": section_text,  # full section — for context enrichment
        "text": text,
    }


# ── Sentence splitting ───────────────────────────────────────────────────

# Abbreviations whose trailing dot is NOT a sentence boundary
_ABBREV = re.compile(
    r"(?:"
    r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|incl"
    r"|Gen|Gov|Sgt|Cpl|Pvt|Col|Capt|Lt|Cmdr|Adm"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    r"|St|Ave|Blvd|Dept|Est|Fig|Eq|Vol|Rev|Ed"
    r"|[A-Z]"                   # single-letter initials: U. S. A.
    r")\Z"
)


def _split_sentences_robust(text: str) -> list[str]:
    """Split text into sentences, handling abbreviations and decimals."""
    # Split on .!? followed by whitespace, but keep the delimiter with the left part
    raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)

    # Re-join fragments that were incorrectly split at abbreviations / decimals
    merged: list[str] = []
    buf = ""
    for frag in raw:
        if buf:
            buf = buf + " " + frag
        else:
            buf = frag
        # Check if buf ends with an abbreviation dot or a decimal dot
        # e.g. "Dr." or "3." (next fragment might start with a digit like "14")
        stripped = buf.rstrip(".")
        if buf.endswith(".") and (
            _ABBREV.search(stripped)
            or re.search(r"\d\Z", stripped)   # trailing digit before dot: "3."
        ):
            continue  # don't flush, keep accumulating
        merged.append(buf.strip())
        buf = ""
    if buf:
        merged.append(buf.strip())

    return [s for s in merged if len(s) > 20]


def _split_sentences(text: str) -> list[str]:
    """Legacy wrapper used by semantic chunker."""
    return _split_sentences_robust(text)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def _hard_split(sentences: list[str], max_chars: int) -> list[list[str]]:
    result, current, length = [], [], 0
    for s in sentences:
        if length + len(s) > max_chars and current:
            result.append(current)
            current, length = [], 0
        current.append(s)
        length += len(s)
    if current:
        result.append(current)
    return result


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (normalised_section_type, section_text)."""
    boundaries: list[tuple[int, str]] = []
    for m in _SECTION_RE.finditer(text):
        title = m.group(0).strip().rstrip(":")
        boundaries.append((m.start(), title))

    if not boundaries:
        return [("body", text)]

    sections = []
    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body = text[start:end].strip()
        sections.append((_normalise_section(title), body))
    return sections


_SECTION_MAP = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "background",
    "method": "methods",
    "experiment": "experiments",
    "result": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "references": "references",
    "appendix": "appendix",
}


def _normalise_section(title: str) -> str:
    t = title.lower()
    for key, val in _SECTION_MAP.items():
        if key in t:
            return val
    return "body"
