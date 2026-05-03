# Dataloader Notes: BEIR NQ and BEIR SciFact

This document studies the NQ and SciFact paths in `dataloader/loader.py`, explains the dataset background, and records the split behavior that matters for this repository's RAG evaluation runs.

Generated: 2026-05-03

## Executive Summary

`load_eval_samples(dataset, split, n=0)` dispatches `dataset="nq"` to `_load_nq` and `dataset="scifact"` to `_load_scifact`. Both loaders normalize BEIR-style retrieval datasets into the shared `EvalSample` schema:

- `id`: BEIR query id.
- `query`: BEIR query text.
- `documents`: a small candidate document pool, not the full corpus.
- `relevant_doc_ids`: positive qrel document ids.
- `reference_answer`: populated for NQ from NQ-Open answer strings when question-text matching succeeds; empty for SciFact.
- `answer_spans`: populated only for NQ when NQ-Open has a matching normalized question.

The most important operational point is that the project does not evaluate over the full BEIR corpus directly. It builds a focused per-query pool:

- NQ: all relevant qrel documents plus the first 500 corpus documents.
- SciFact: all relevant qrel documents plus the first 300 corpus documents.

This makes experiments lighter and ensures the relevant document is present, but it also changes the retrieval problem from full-corpus BEIR retrieval to reranking/retrieval over a curated candidate set.

## Loader Entry Point

Relevant code: `dataloader/loader.py`

- Lines 20-34 define `load_eval_samples()` and the supported dataset keys: `nq`, `scifact`, `qasper`, and `quality`.
- Lines 41-60 implement sample-level caching in `CACHE_DIR/samples`, defaulting to `./cache/samples`.
- Cache file naming is `{dataset}_{split}.pkl`, for example `cache/samples/nq_test.pkl`.
- If a cache exists, the loader returns cached samples and only applies `n` slicing after load.
- If no cache exists, it loads from Hugging Face datasets, writes the cache, then applies `n`.

Local cache observations in this workspace:

| Cache file | Samples | Notes |
| --- | ---: | --- |
| `cache/samples/nq_test.pkl` | 3,452 | First 10 cached samples each have 500 docs; relevant docs per sample range 1-4; 2,255 samples have matched NQ-Open answer spans. |
| `cache/samples/scifact_test.pkl` | 300 | First 10 cached samples have about 300-302 docs; relevant docs per sample range 1-5; answer spans are not populated. |

## BEIR Format Background

BEIR is a heterogeneous benchmark for zero-shot information retrieval. BEIR datasets use three logical files:

- `corpus`: documents/passages, with `_id`, optional `title`, and `text`.
- `queries`: query ids and query text.
- `qrels`: relevance judgments with `query-id`, `corpus-id`, and integer `score`.

In this repository, positive qrels are rows where `score > 0`. All positive corpus ids for a query become `EvalSample.relevant_doc_ids`.

Sources:

- BEIR / NQ dataset card: https://huggingface.co/datasets/BeIR/nq
- BEIR / NQ qrels dataset card: https://huggingface.co/datasets/BeIR/nq-qrels
- BEIR / SciFact dataset card: https://huggingface.co/datasets/BeIR/scifact
- BEIR / SciFact qrels dataset card: https://huggingface.co/datasets/BeIR/scifact-qrels
- BEIR paper citation page in dataset cards: https://openreview.net/forum?id=wCu6T5xFjeJ

## NQ (BEIR)

### Dataset Background

Natural Questions (NQ) is a Google Research question-answering dataset built from real anonymized search queries and Wikipedia answer annotations. The original NQ task presents a question with a Wikipedia page and asks systems to identify long and short answers when present. Google Research reports the public release as 307,373 training examples, 7,830 development examples, and 7,842 sequestered test examples.

BEIR's NQ version adapts Natural Questions into an information-retrieval task: retrieve Wikipedia passages/documents relevant to a natural-language question. The BEIR dataset card lists NQ under question-answering retrieval with 3,452 queries, a 2.68M-document corpus, and about 1.2 relevant documents per query.

This loader also consults `google-research-datasets/nq_open`, a separate open-domain QA version derived from Natural Questions. NQ-Open provides answer strings rather than BEIR qrels. TensorFlow Datasets lists NQ-Open as 87,925 train examples and 3,610 validation examples, with fields `question` and `answer`.

Sources:

- Google Research Natural Questions publication page: https://research.google/pubs/natural-questions-a-benchmark-for-question-answering-research/
- Google Research Natural Questions repository: https://github.com/google-research-datasets/natural-questions
- NQ-Open TFDS page: https://www.tensorflow.org/datasets/catalog/natural_questions_open
- BEIR / NQ Hugging Face page: https://huggingface.co/datasets/BeIR/nq
- BEIR / NQ qrels Hugging Face page: https://huggingface.co/datasets/BeIR/nq-qrels

### Loader Behavior

Code path: `_load_nq(split)` at `dataloader/loader.py` lines 65-123.

1. Loads the BEIR NQ corpus:
   - `load_dataset("BeIR/nq", "corpus", split="corpus")`
   - HF page reports about 2.68M corpus rows.

2. Loads all BEIR NQ queries:
   - `load_dataset("BeIR/nq", "queries", split="queries")`
   - HF page reports about 3.45k query rows.

3. Maps the requested project split to a qrels split:
   - `train -> train`
   - `dev -> validation`
   - `test -> test`
   - any other value is passed through unchanged.

4. Loads qrels:
   - `load_dataset("BeIR/nq-qrels", split=qrel_split)`
   - Current HF `BeIR/nq-qrels` exposes a single `test` split with about 4.2k qrel rows. This means `split="test"` is the reliable split for the current loader/dataset combination. `train` or `dev` will request qrel splits that are not exposed by the HF qrels dataset card.

5. Loads answer strings from NQ-Open:
   - `_load_nq_answers(split)` maps `train -> train`, `dev -> validation`, and `test -> validation`.
   - It lowercases and strips each NQ-Open question, storing `question -> list[str answers]`.
   - Matching is exact after `strip().lower()`; punctuation, wording, or tokenization differences prevent answer matching.
   - If loading NQ-Open fails, answers are optional and the loader continues with an empty answer map.

6. Builds `doc_map` from BEIR corpus id to text only:
   - `doc_map = {str(row["_id"]): row["text"] for row in corpus_ds}`
   - The BEIR title is discarded.

7. Builds `qrel_map`:
   - Keeps only qrel rows with `score > 0`.
   - Stores `query-id -> list[corpus-id]`.

8. Iterates over all queries:
   - Skips any query id with no qrel entry for the chosen split.
   - Uses `relevant_ids` from qrels.
   - Builds a focused candidate pool: `set(relevant_ids) union first 500 corpus ids`.
   - Converts pool ids into `{"id": did, "text": doc_map[did]}` documents.
   - Looks up NQ-Open answers by normalized query text.
   - Emits `EvalSample`.

### NQ Split Information

Published / upstream split information:

| Source | Split information |
| --- | --- |
| Original Natural Questions | 307,373 train; 7,830 dev; 7,842 hidden/sequestered test. |
| NQ-Open | 87,925 train; 3,610 validation. |
| BEIR NQ dataset card | BEIR NQ is listed as train/test in the BEIR direct-download table; corpus has about 2.68M rows; queries have about 3.45k rows. |
| HF `BeIR/nq-qrels` | Current dataset viewer exposes only `test`, about 4.2k qrel rows. |

Repository split behavior:

| Requested `split` | BEIR qrels requested | NQ-Open answers requested | Expected result with current HF qrels |
| --- | --- | --- | --- |
| `train` | `train` | `train` | Likely fails unless a `train` qrels split is available in the local HF cache or dataset revision. |
| `dev` | `validation` | `validation` | Likely fails unless a `validation` qrels split is available in the local HF cache or dataset revision. |
| `test` | `test` | `validation` | Works with current HF qrels; local cache has 3,452 samples. |

### NQ Evaluation Meaning in This Repo

For NQ, each sample is a retrieval-plus-answering example:

- Retrieval labels come from BEIR qrels.
- Reference answer strings come from NQ-Open, not BEIR.
- `reference_answer` is the first matched NQ-Open answer string.
- `answer_spans` contains all matched valid answer strings.
- If no NQ-Open match is found, `reference_answer=""` and `answer_spans=[]`, even though relevant documents still exist.

Because the document pool includes the positive docs by construction, retrieval metrics measure ranking inside the curated pool rather than the harder full 2.68M-corpus BEIR task.

## SciFact (BEIR)

### Dataset Background

SciFact is a scientific claim verification dataset introduced by Wadden et al., "Fact or Fiction: Verifying Scientific Claims." The original task asks a system to retrieve abstracts from the scientific literature, decide whether they support or refute a claim, and identify rationale sentences. The original AllenAI dataset contains expert-written scientific claims, evidence-containing abstracts, labels such as SUPPORT/REFUTE, and sentence-level rationales.

The AllenAI Hugging Face dataset card lists:

- Claims: 1,261 train, 450 validation, 300 test.
- Corpus: 5,183 evidence documents.

The BEIR adaptation turns SciFact into a retrieval task: given a scientific claim as a query, retrieve abstracts/documents that contain evidence. BEIR's direct-download table lists SciFact as train/test with 300 queries, about 5K corpus documents, and about 1.1 relevant documents per query. The BEIR `scifact` HF page exposes `corpus` and `queries` subsets; its dataset card reports 5,183 corpus rows and 1,109 query rows.

Sources:

- AllenAI SciFact repository: https://github.com/allenai/scifact
- AllenAI SciFact HF dataset: https://huggingface.co/datasets/allenai/scifact
- BEIR / SciFact HF dataset: https://huggingface.co/datasets/BeIR/scifact
- BEIR / SciFact qrels HF dataset: https://huggingface.co/datasets/BeIR/scifact-qrels
- SciFact paper: https://aclanthology.org/2020.emnlp-main.609/

### Loader Behavior

Code path: `_load_scifact(split)` at `dataloader/loader.py` lines 146-190.

1. Loads the BEIR SciFact corpus:
   - `load_dataset("BeIR/scifact", "corpus", split="corpus")`
   - HF page reports 5,183 corpus rows.

2. Loads all BEIR SciFact queries:
   - `load_dataset("BeIR/scifact", "queries", split="queries")`
   - HF page reports 1,109 query rows.

3. Maps the requested project split to a qrels split:
   - `train -> train`
   - `dev -> validation`
   - `test -> test`
   - any other value is passed through unchanged.

4. Loads qrels:
   - First tries `load_dataset("BeIR/scifact-qrels", split=qrel_split)`.
   - If any exception occurs, falls back to `load_dataset("BeIR/scifact-qrels", split="test")`.
   - This broad fallback means `split="dev"` can silently become test qrels when `validation` is unavailable.

5. Builds `doc_map` from BEIR corpus id to text only:
   - `doc_map = {str(row["_id"]): row["text"] for row in corpus_ds}`
   - The title is discarded.

6. Builds `qrel_map`:
   - Keeps rows with `score > 0`.
   - Stores `query-id -> list[corpus-id]`.

7. Iterates over all queries:
   - Skips query ids not present in the selected qrels.
   - Uses positive qrels as `relevant_doc_ids`.
   - Builds a focused pool: `set(relevant_ids) union first 300 corpus ids`.
   - Emits `EvalSample` with `reference_answer=""`.

### SciFact Split Information

Published / upstream split information:

| Source | Split information |
| --- | --- |
| Original AllenAI SciFact | Claims: 1,261 train, 450 validation, 300 test; corpus: 5,183 documents. Original repo notes train/dev contain labels, while test labels are not public for original leaderboard evaluation. |
| BEIR SciFact dataset card | Corpus: 5,183 rows; queries: 1,109 rows. BEIR direct-download table lists SciFact as train/test, 300 test queries, about 5K corpus, 1.1 relevant docs/query. |
| HF `BeIR/scifact-qrels` | Current qrels page exposes qrels separately from the corpus/query dataset. The search/dataset card indicates at least train and test qrels; if `validation` is not present, the loader falls back to test. |

Repository split behavior:

| Requested `split` | BEIR qrels requested | Fallback behavior | Expected result |
| --- | --- | --- | --- |
| `train` | `train` | fallback to `test` on any exception | Should load train qrels if available; otherwise silently uses test. |
| `dev` | `validation` | fallback to `test` on any exception | Likely uses test if no validation qrels split exists. |
| `test` | `test` | fallback to `test` on any exception | Works; local cache has 300 samples. |

### SciFact Evaluation Meaning in This Repo

For SciFact, each sample is a claim-to-evidence retrieval example:

- Query text is the scientific claim.
- Relevant document ids are evidence abstracts from BEIR qrels.
- The loader does not expose SUPPORT/REFUTE labels.
- The loader does not expose rationale sentences.
- `reference_answer` is always empty, so answer-generation metrics that require a textual gold answer should not run or should be interpreted as unavailable.

This is appropriate for retrieval and reranking experiments, but not for full SciFact claim-verification evaluation.

## Differences Between Published Tasks and This Loader

| Area | Published dataset/task | This loader |
| --- | --- | --- |
| NQ task | QA over Wikipedia / open-domain answer prediction; BEIR adaptation is full-corpus retrieval. | Retrieval samples over a curated pool plus optional NQ-Open answer strings matched by question text. |
| NQ corpus | BEIR corpus has about 2.68M rows. | Each sample gets relevant docs plus first 500 corpus docs. |
| NQ answer labels | NQ-Open has answer strings for train/validation. | `test` uses NQ-Open validation answers for string matching. Only exact normalized question matches populate answers. |
| SciFact task | Retrieve evidence abstracts, classify SUPPORT/REFUTE, select rationale sentences. | Retrieval only; no stance labels or rationales are kept. |
| SciFact corpus | 5,183 evidence documents. | Each sample gets relevant docs plus first 300 corpus docs. |
| Split handling | Dataset-specific upstream split availability. | Project accepts `train/dev/test`, but BEIR qrels availability is uneven; SciFact catches errors and falls back to test, NQ does not. |

## Result-Related Notes

The loader details above explain what data the project evaluates. The current result summaries live in separate documents:

- `docs/nq_400_ablation_summary.md` covers the Natural Questions ablation runs. In those runs, NQ retrieval was already strong, and reranking was the main source of improvement.
- `docs/scifact_300_ablation_summary.md` covers the SciFact ablation runs. SciFact should be described as 300 samples, not 400, because the completed local SciFact runs loaded 300 qrel-backed BEIR/SciFact test samples.

The main SciFact result finding is dataset-specific: `sentence chunking + BAAI/bge-large-en-v1.5 + dense retrieval + no reranker` was the strongest tested local configuration. It improved over `hybrid + cross-encoder/ms-marco-MiniLM-L-6-v2` primarily by changing:

- `retrieval_mode: hybrid -> dense`
- `reranker: cross-encoder -> none`

That change raised the local SciFact result from `MRR@1 = 0.7800`, `MRR@10 = 0.8345`, `Recall@10 = 0.9027` to `MRR@1 = 0.8100`, `MRR@10 = 0.8646`, `Recall@10 = 0.9550`.

These result documents should still be read with the loader caveat in mind: the experiments use the project's per-query candidate pools, not official full-corpus BEIR retrieval.

## Practical Recommendations

1. Prefer `DATASET_SPLIT=test` for both `nq` and `scifact` unless the local HF dataset revisions are known to expose the requested qrel split.

2. Treat retrieval metrics as candidate-pool metrics, not official BEIR full-corpus metrics.

3. For NQ answer metrics, report how many samples have non-empty `answer_spans`. In the current local `nq_test` cache, 2,255 of 3,452 samples have matched answer spans.

4. For SciFact, do not interpret empty `reference_answer` as a negative/no-answer label. It means this loader has not imported the original claim-verification labels.

5. If official BEIR comparability is desired, change the retrieval pipeline to index/search the full corpus rather than injecting relevant documents into each sample pool.

6. If full SciFact verification is desired, load `allenai/scifact` claims and preserve `evidence_label` plus `evidence_sentences` in an extended schema.

7. Consider tightening split error handling:
   - For NQ, explicitly validate that the requested qrels split exists and give a clear error.
   - For SciFact, avoid silently falling back from `dev` to `test`; log or raise a split-specific warning/error.

## Citation Pointers

- BEIR: Nandan Thakur, Nils Reimers, Andreas Rueckle, Abhishek Srivastava, Iryna Gurevych. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." NeurIPS Datasets and Benchmarks, 2021. https://openreview.net/forum?id=wCu6T5xFjeJ
- Natural Questions: Tom Kwiatkowski et al. "Natural Questions: A Benchmark for Question Answering Research." TACL, 2019. https://research.google/pubs/natural-questions-a-benchmark-for-question-answering-research/
- NQ-Open: Derived open-domain QA benchmark from Natural Questions, documented by TensorFlow Datasets. https://www.tensorflow.org/datasets/catalog/natural_questions_open
- SciFact: David Wadden et al. "Fact or Fiction: Verifying Scientific Claims." EMNLP, 2020. https://aclanthology.org/2020.emnlp-main.609/
