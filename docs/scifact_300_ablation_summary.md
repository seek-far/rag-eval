# SciFact Ablation Summary (300 samples)

This ablation compares retrieval and reranking choices on the loaded SciFact test set. Although the run config banner may show `200 + 200`, BEIR/SciFact test provides 300 qrel-backed samples in this project, so the completed SciFact runs report `n_samples = 300`.

**Shared setup**
- Embedding model: `BAAI/bge-large-en-v1.5`
- Main chunking setting: `sentence` or `semantic`
- Compared retrieval modes: `hybrid` and `dense`
- Compared rerankers: `none`, MS MARCO MiniLM cross-encoders, `BAAI/bge-reranker-large`, `ncbi/MedCPT-Cross-Encoder`, `NeuML/biomedbert-base-reranker`, and an OpenAI-compatible LLM cross-encoder using `deepseek-v4-pro`

![SciFact ablation figure](figures/scifact_300_ablation.png)

## Headline

SciFact tells a different story from Natural Questions: generic web-trained rerankers underperform here, but biomedical cross-encoders and LLM-based reranking help. The best completed single-model top-1 run is `sentence + dense retrieval + DeepSeek LLM cross-encoder`, which reaches `MRR@1 = 0.8500`, narrowly ahead of `MedCPT` at `MRR@1 = 0.8467`.

Offline replay over saved run artifacts suggests score-level and learned fusion can improve the retained ranking without rerunning retrieval, reranking, or LLM API calls. A fixed test-set-tuned fusion reached `MRR@1 = 0.8600`, `MRR@10 = 0.9033`, and `Recall@5 = 0.9550`, but the cleaner 5-fold fixed-weight CV replay reached `MRR@1 = 0.8534`, `MRR@10 = 0.8975`, and `Recall@5 = 0.9500`. A 5-fold learned top-1 fusion replay did better, reaching `MRR@1 = 0.8600`, `MRR@10 = 0.9059`, and `Recall@5 = 0.9572`.

## Key takeaways

1. **Domain-matched cross-encoders are the strongest SciFact rerankers.**
   - `sentence + dense + MedCPT CE` improves over dense retrieval without reranking by `+0.0367` on `MRR@1` and `+0.0268` on `MRR@10`.
   - `sentence + dense + BiomedBERT CE` also improves over the dense baseline by `+0.0300` on `MRR@1` and `+0.0186` on `MRR@10`.
   - This suggests the reranking issue was model/domain fit, not reranking itself.

2. **The LLM cross-encoder gives the best MRR@1, but not the best coverage.**
   - `sentence + dense + DeepSeek LLM CE` reaches `MRR@1 = 0.8500`, the highest top-1 score in this table.
   - Its `Recall@5 = 0.9243` is lower than `MedCPT` (`0.9402`) and dense retrieval without reranking (`0.9310`).
   - This makes the LLM CE useful when the objective is specifically top-1 ranking, but less attractive if top-5 coverage is the main target.

3. **Cross-encoder reranking is not universally helpful.**
   - Both `sentence + hybrid + cross-encoder (MiniLM)` and `semantic + hybrid + cross-encoder (MiniLM)` land at roughly `MRR@1 = 0.7800` and `MRR@10 = 0.834`.
   - `sentence + dense + BGE reranker large` reaches only `MRR@1 = 0.7700`, and `sentence + dense + MiniLM-L12` reaches only `MRR@1 = 0.7600`.
   - This is the opposite of the NQ pattern, where a general MS MARCO reranker was a clear win.

4. **Chunking matters less than retrieval/reranking choice here too.**
   - Under MiniLM cross-encoder reranking, `sentence` and `semantic` chunking are nearly identical:
   - `sentence + hybrid + CE`: `MRR@1 = 0.7800`, `MRR@10 = 0.8345`
   - `semantic + hybrid + CE`: `MRR@1 = 0.7800`, `MRR@10 = 0.8344`
   - The meaningful differences come from retrieval mode and reranker behavior, not chunk boundaries.

5. **Offline score fusion helps coverage, but fixed global weights are not a full answer.**
   - The best score-fusion replay used mostly MedCPT with smaller LLM CE, BGE, BiomedBERT, and dense-score contributions.
   - It improved the retained ranking to `MRR@1 = 0.8600`, `MRR@10 = 0.9033`, and `Recall@5 = 0.9550`.
   - A 5-fold fixed-weight optimization replay was more conservative: `MRR@1 = 0.8534`, `MRR@10 = 0.8975`, `Recall@5 = 0.9500`.
   - This means simple fusion is promising for `MRR@10` and `Recall@5`, while robust `MRR@1` gains likely need learned top-1 selection rather than one global weight vector.

6. **Learned top-1 fusion improves over fixed fusion in cross-validation.**
   - A new offline learner built `4,537` candidate rows from the saved artifacts, with `329` positive rows across `300` samples.
   - It used about two-thirds of the 16-core machine (`10` workers) for parallel tree models.
   - Logistic regression, random forest, and extra-trees all reached `MRR@1 = 0.8600` under 5-fold query-grouped cross-validation.
   - Random forest gave the strongest overall ranking: `MRR@10 = 0.9059`, `Recall@5 = 0.9572`.

## Compact results table

| Configuration | MRR@1 | MRR@10 | Hit@5 | Recall@5 |
|---|---:|---:|---:|---:|
| Sentence + dense + LLM CE (DeepSeek, 50 workers) | 0.8500 | 0.8872 | 0.9333 | 0.9243 |
| Sentence + dense + CE (MedCPT) | 0.8467 | 0.8914 | 0.9467 | 0.9402 |
| Sentence + dense + CE (BiomedBERT) | 0.8400 | 0.8832 | 0.9433 | 0.9367 |
| Sentence + dense + none | 0.8100 | 0.8646 | 0.9333 | 0.9310 |
| Sentence + dense + CE (BGE reranker large) | 0.7700 | 0.8388 | 0.9236 | 0.9236 |
| Sentence + dense + CE (MiniLM-L12) | 0.7600 | 0.8219 | 0.9133 | 0.9038 |
| Sentence + hybrid + CE (MiniLM-L6) | 0.7800 | 0.8345 | 0.9133 | 0.9027 |
| Semantic + hybrid + CE (MiniLM-L6) | 0.7800 | 0.8344 | 0.9133 | 0.9027 |

## Interpretation

The most plausible explanation is that SciFact needs a reranker that understands biomedical/scientific abstracts rather than only general web passage relevance. The MS MARCO MiniLM rerankers and BGE reranker are good general candidates, but they can reward topical similarity without reliably capturing scientific evidence fit, negation, directionality, or biomedical entity specificity.

The MedCPT and BiomedBERT results change the earlier conclusion: reranking can help SciFact, but only when the reranker is domain-matched. The DeepSeek LLM CE pushes `MRR@1` slightly higher than MedCPT, but MedCPT remains a better balance of top-1 quality, top-5 coverage, speed, and operational simplicity.

## Failure Overlap

The saved artifacts include `per_sample_results.jsonl`, so failure overlap was analyzed without rerunning the pipeline. A top-1 failure means the relevant document was not ranked first.

| Model | Top-1 failures | Top-5 misses |
|---|---:|---:|
| MedCPT | 46 | 22 |
| BiomedBERT | 48 | 23 |
| BGE reranker large | 69 | 27 |
| LLM CE | 45 | 27 |

Across the four rerankers, `90` samples failed top-1 for at least one model, but only `23` failed top-1 for all four. The overlap is therefore real but not dominant. MedCPT and BiomedBERT are the most similar pair, sharing `38` top-1 failures with a Jaccard overlap of `0.679`; BGE large and LLM CE are the least similar pair, sharing `32` top-1 failures with a Jaccard overlap of `0.390`.

The important diagnostic split is whether a top-1 failure still retained the relevant document in the final reranked candidates:

| Model | Top-1 fail but relevant still retained | Relevant not retained |
|---|---:|---:|
| MedCPT | 30 | 16 |
| BiomedBERT | 31 | 17 |
| BGE reranker large | 49 | 20 |
| LLM CE | 25 | 20 |

There are also `9` shared hard misses where the relevant document was absent from the pre-rerank candidates for all four rerankers. These cannot be fixed by cross-encoder fusion alone; they need retrieval changes or a larger rerank candidate pool.

## Offline Fusion Replay

Fusion was tested by replaying the saved detailed outputs from:

- `2026-05-06T21_39_02_exp_013_dense_ce_medcpt`
- `2026-05-06T21_40_18_exp_014_dense_ce_biomedbert`
- `2026-05-06T21_43_43_exp_015_dense_ce_bge_large`
- `2026-05-06T22_56_43_exp_019_dense_llm_ce_deepseek_workers50_retry_parse`

This replay did not rerun retrieval, indexing, local cross-encoders, or the LLM API. It only combined stored `retrieval_score`, stored `rerank_score`, final candidates, and relevance labels.

Rank-only reciprocal-rank fusion was a poor fit for top-1 ranking. Equal-weight RRF improved retained coverage in some cases but dropped `MRR@1` to roughly `0.68-0.71`, because consensus topical false positives were promoted above the exact evidence document.

Score-level fusion was much better. The best test-set-tuned replay used min-max normalized scores with:

`MedCPT 0.70 + BiomedBERT 0.05 + BGE large 0.05 + LLM CE 0.20 + dense score 0.20`

| Method | MRR@1 | MRR@10 | Hit@5 | Recall@5 |
|---|---:|---:|---:|---:|
| Best single run: LLM CE | 0.8500 | 0.8872 | 0.9333 | 0.9243 |
| Best single local CE: MedCPT | 0.8467 | 0.8914 | 0.9467 | 0.9402 |
| Offline score fusion, test tuned | 0.8600 | 0.9033 | 0.9567 | 0.9550 |
| Offline fixed score fusion, 5-fold selected | 0.8534 | 0.8975 | 0.9500 | 0.9500 |
| Oracle top-1 over four reranker outputs | 0.9233 | 0.9372 | 0.9600 | 0.9583 |

The oracle row means that, for many samples, one reranker already placed the relevant document first even when others did not. The gap between score fusion and the oracle motivates the next line of work: learned top-1 reranking/fusion, ideally trained on held-out data rather than tuned directly on the 300-sample SciFact test set.

The fixed-weight 5-fold CV used a `0.05` grid over the four cross-encoder weights plus dense-score weights from `0, 0.05, 0.1, 0.2, 0.3, 0.5`, for `10,602` combinations per fold. The run used `10` workers on the 16-core machine. Selected weights varied by fold, which suggests fixed global weights are somewhat unstable on only 300 queries.

## Learned Top-1 Fusion Replay

The first learned-fusion experiment is implemented in `utils/offline_learned_fusion.py`. It is still offline: it reads the saved artifact files, creates one row per `(query, candidate_doc)`, and trains small classifiers to predict whether a candidate document is relevant.

The feature table includes:

- dense retrieval rank and score
- per-reranker presence, rank, raw score, and normalized score for MedCPT, BiomedBERT, BGE large, and LLM CE
- model agreement features such as how many rerankers retained the candidate and how many ranked it first

The run used `10` workers on a 16-core machine and 5-fold `GroupKFold`, grouping by query so candidates from the same query do not appear in both train and validation folds. Output was saved under:

`results/offline_fusion/2026-05-07T00_02_48_learned_top1_fusion`

| Method | MRR@1 | MRR@10 | Hit@5 | Recall@5 |
|---|---:|---:|---:|---:|
| LLM CE baseline | 0.8500 | 0.8872 | 0.9333 | 0.9243 |
| MedCPT baseline | 0.8467 | 0.8914 | 0.9467 | 0.9402 |
| Fixed score fusion, test tuned | 0.8600 | 0.9033 | 0.9567 | 0.9550 |
| Fixed score fusion, 5-fold selected | 0.8534 | 0.8975 | 0.9500 | 0.9500 |
| Learned fusion: logistic regression, 5-fold | 0.8600 | 0.9024 | 0.9567 | 0.9543 |
| Learned fusion: random forest, 5-fold | 0.8600 | 0.9059 | 0.9600 | 0.9572 |
| Learned fusion: extra trees, 5-fold | 0.8600 | 0.9050 | 0.9600 | 0.9559 |
| Oracle top-1 over four reranker outputs | 0.9233 | 0.9372 | 0.9600 | 0.9583 |

The learned models match the best fixed-fusion top-1 result while improving `MRR@10` and `Recall@5`. The remaining gap to the oracle suggests that better features, more training data, or a ranking-specific objective may still extract more of the available complementarity.

## LLM CE Timing

The LLM CE experiment used an OpenAI-compatible endpoint with `deepseek-v4-pro` during an idle period in China local time. The run was executed on May 6, 2026 around 22:49-22:56 Europe/Berlin time, corresponding to early morning on May 7, 2026 in China, when API traffic may be lower than daytime business hours.

Observed timings:

| Setting | Outcome | Wall time | Notes |
|---|---:|---:|---|
| 1 sample, 1 request | success | ~39.1s | Baseline single-request latency |
| 2 samples, 2 workers | success | ~49.2s | Both calls overlapped; about 1.6x faster than the sequential estimate |
| 300 samples, 10 workers | failed | ~24.9 min | Failed after 38/300 completed due to an LLM read timeout before timeout retry handling was broadened |
| 300 samples, 50 workers | failed first, then success after parser retry handling | 7.86 min | Final successful run completed all 300 samples |

Using the 1-sample baseline, a sequential 300-sample run would be roughly `300 * 39.1s = 195.7 minutes`. The successful 50-worker run completed in `7.86 minutes`, an approximate `24.9x` wall-clock speedup. The LLM reranking phase itself took about `7m36s`, giving a similar rerank-only speedup of about `25.8x`.

The 50-worker run also produced multiple empty, truncated, or malformed JSON responses from the LLM endpoint. The final successful run depended on rainy-day handling: retrying request timeouts and retrying unparsable score payloads. This is important for production-style experiments because high concurrency improves throughput but increases the chance of transient bad responses or long-tail latency.

## Recommendation

For SciFact, the best default from this set of runs is:

`sentence chunking + dense retrieval + ncbi/MedCPT-Cross-Encoder`

If the only goal is maximum `MRR@1` and external API latency/cost are acceptable, `sentence chunking + dense retrieval + llm-cross-encoder` with `deepseek-v4-pro` is the best observed top-1 scorer. If latency, cost, determinism, or top-5 coverage matter, `MedCPT` remains the stronger default. If reranker latency is unacceptable, `sentence chunking + dense retrieval + no reranker` remains a solid baseline.

For fusion, prefer score-level replay or learned top-1 selection over RRF. The learned fusion replay is now the best offline candidate from this set, but the stable production candidate still needs validation on a held-out split or cross-dataset transfer.

## Caveats

- These conclusions are based on the loaded `300`-sample SciFact test set used by the completed local runs.
- Not every row varies just one factor, so this is a practical ablation rather than a perfectly controlled study.
- Reranked runs keep only `RERANK_TOP_K = 5` before metric computation, so `MRR@10`, `Hit@10`, and `Recall@10` in those runs should be interpreted as metrics over the retained reranked set. `MRR@1` and `Recall@5` are the clearest comparisons here.
- Offline fusion only replays candidates and scores that were already saved. It cannot recover candidates that were not retrieved or not retained by the original runs.
- The best offline score-fusion row is tuned on the same 300 SciFact samples, so it is an optimistic diagnostic; the 5-fold fixed-weight and learned-fusion replays are more conservative estimates.
- LLM CE timing depends heavily on endpoint location, model behavior, provider load, and time of day. The reported 50-worker timing was measured during an idle period in China local time and may not hold during peak hours.
- High-concurrency LLM CE runs need rainy-day handling for timeouts, empty responses, and malformed JSON. In this repo, `LLM_RERANK_RETRIES`, `LLM_RERANK_RETRY_SLEEP`, `LLM_TIMEOUT`, and JSON parser retries are part of that safety net.
- The best SciFact run here is `dense + biomedical cross-encoder`, while the NQ best run was `hybrid + general MS MARCO cross-encoder`; that contrast is a useful reminder that retrieval defaults may need to be dataset-specific.
