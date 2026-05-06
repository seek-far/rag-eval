# SciFact Ablation Summary (300 samples)

This ablation compares retrieval and reranking choices on the loaded SciFact test set. Although the run config banner may show `200 + 200`, BEIR/SciFact test provides 300 qrel-backed samples in this project, so the completed SciFact runs report `n_samples = 300`.

**Shared setup**
- Embedding model: `BAAI/bge-large-en-v1.5`
- Main chunking setting: `sentence` or `semantic`
- Compared retrieval modes: `hybrid` and `dense`
- Compared rerankers: `none`, MS MARCO MiniLM cross-encoders, `BAAI/bge-reranker-large`, `ncbi/MedCPT-Cross-Encoder`, `NeuML/biomedbert-base-reranker`, and an OpenAI-compatible LLM cross-encoder using `deepseek-v4-pro`

![SciFact ablation figure](figures/scifact_300_ablation.png)

## Headline

SciFact tells a different story from Natural Questions: generic web-trained rerankers underperform here, but biomedical cross-encoders and LLM-based reranking help. The best top-1 run is now `sentence + dense retrieval + DeepSeek LLM cross-encoder`, which reaches `MRR@1 = 0.8500`, narrowly ahead of `MedCPT` at `MRR@1 = 0.8467`.

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

## Caveats

- These conclusions are based on the loaded `300`-sample SciFact test set used by the completed local runs.
- Not every row varies just one factor, so this is a practical ablation rather than a perfectly controlled study.
- Reranked runs keep only `RERANK_TOP_K = 5` before metric computation, so `MRR@10`, `Hit@10`, and `Recall@10` in those runs should be interpreted as metrics over the retained reranked set. `MRR@1` and `Recall@5` are the clearest comparisons here.
- LLM CE timing depends heavily on endpoint location, model behavior, provider load, and time of day. The reported 50-worker timing was measured during an idle period in China local time and may not hold during peak hours.
- High-concurrency LLM CE runs need rainy-day handling for timeouts, empty responses, and malformed JSON. In this repo, `LLM_RERANK_RETRIES`, `LLM_RERANK_RETRY_SLEEP`, `LLM_TIMEOUT`, and JSON parser retries are part of that safety net.
- The best SciFact run here is `dense + biomedical cross-encoder`, while the NQ best run was `hybrid + general MS MARCO cross-encoder`; that contrast is a useful reminder that retrieval defaults may need to be dataset-specific.
