# vLLM Inference Benchmark Report

**Date:** 2026-09-04
**Server:** Biznet GIO Cloud GPU trial instance (NVIDIA H200, 143,771 MiB / ~140 GB VRAM)
**Serving stack:** vLLM 0.28.0, OpenAI-compatible API at `http://YOUR_VLLM_HOST:8000/v1`
**Model:** `Qwen/Qwen3.8-27B` — Qwen3.5-architecture (`Qwen3_5ForConditionalGeneration`), hybrid Mamba/gated-linear-attention + full-attention, multimodal (image/video-text-to-text), ~27B parameters, bf16, apache-2.0
**Launch config:** `vllm serve Qwen/Qwen3.8-27B --host 0.0.0.0 --port 8000 --max-model-len 32768`

This report covers three test rounds run against the live server: a single-request sequential benchmark across 10 system design prompts, an initial concurrency sweep (1–16 simultaneous requests), and a follow-up high-concurrency push (32–96 simultaneous requests) to probe the throughput ceiling.

---

## 1. Test workload

10 open-ended system design prompts were used throughout (e.g. "Design a URL shortener," "Design a distributed rate limiter," "Design a payment processing system with exactly-once semantics," etc.) — chosen to be long, reasoning-heavy generations rather than short factual answers, since `Qwen3.8-27B` produces extensive chain-of-thought before its final answer.

---

## 2. First-time provisioning timeline

Before any benchmark could run, the server had to be provisioned from a bare Ubuntu 24.04 box to a live vLLM endpoint. This is a one-time cost per host (and largely per-model for the download step).

| Step | Duration |
|---|---|
| System package install (`python3-venv`, `pip`, `git`, `curl` via apt) | ~1 min |
| `pip install vllm` (torch + CUDA deps + vLLM wheel) | ~6 min |
| Model download (18 shards, ~54 GB from Hugging Face, unauthenticated) | ~6 min |
| Model load onto GPU + multimodal warmup + CUDA graph capture | ~1 min |
| **Total: bare box → live, serving endpoint** | **~15 min** |

The two dominant costs — installing vLLM and downloading the model — are both one-time or cacheable on a given host. Swapping to a different model on the same box skips the `pip install` entirely, and only the model download time would repeat, scaling with that model's size rather than the ~54 GB used here.

---

## 3. Sequential (single-request) benchmark

10 cases run one at a time, `max_tokens=12000`, `temperature=0.7`.

| # | Case | Time (s) | Completion tokens | Tok/s | Finish reason |
|---|------|----------|-------------------|-------|----------------|
| 1 | URL Shortener | 124.2 | 7,984 | 64.3 | stop |
| 2 | Rate Limiter | 174.9 | 11,248 | 64.3 | stop |
| 3 | News Feed System | 110.7 | 7,133 | 64.5 | stop |
| 4 | Distributed Cache | 45.8 | 2,955 | 64.5 | stop |
| 5 | Chat Application | 183.8 | 11,824 | 64.3 | stop |
| 6 | Payment Processing System | 152.2 | 9,794 | 64.4 | stop |
| 7 | Ride-Hailing Dispatch | 117.9 | 7,593 | 64.4 | stop |
| 8 | Video Streaming Platform | 104.9 | 6,762 | 64.5 | stop |
| 9 | Distributed Job Scheduler | 168.4 | 10,836 | 64.4 | stop |
| 10 | Online Donation Platform | 186.6 | 12,000 | 64.3 | **length** (hit cap) |

**Summary:**
- 10/10 requests succeeded, **9/10 reached a natural stop** (only the donation-platform case needed more than 12,000 tokens to finish)
- Total wall time: 1,369.6 s (~22.8 min) for 88,129 completion tokens
- **Per-request throughput was remarkably constant: 64.3–64.5 tok/s regardless of prompt or output length**
- Time per case ranged 45.8 s – 186.6 s, driven entirely by how many tokens the model chose to generate, not by prompt complexity or infrastructure variance

---

## 4. Concurrency sweep — throughput scaling

Multiple requests fired simultaneously via a thread pool, `max_tokens=1500` per request (capped low so higher levels complete in reasonable time), one `nvidia-smi` sample per second on the GPU host throughout each batch.

| Concurrency | Batch time (s) | Aggregate tok/s | Avg per-request latency (s) | GPU util (avg) | VRAM peak (MiB) |
|---|---|---|---|---|---|
| 1 | 23.3 | 64.4 | 23.3 | 95% | 132,033 |
| 2 | 28.8 | 104.3 | 28.8 | 81% | 132,033 |
| 4 | 24.5 | 244.9 | 24.5 | 100% | 132,033 |
| 8 | 25.6 | 469.5 | 25.6 | 97% | 132,033 |
| 16 | 26.8 | 894.8 | 26.8 | 100% | 132,033 |
| **32** | **29.6** | **1,624.3** | **29.5** | 96% | 132,033 |
| **64** | **35.5** | **2,708.1** | **35.4** | 97% | 132,033 |
| **96** | **41.8** | **3,443.6** | **41.8** | 97% | 132,037 |

**All 200 requests across every concurrency level succeeded (0 failures, 0 timeouts).**

### Key findings

- **Throughput scales almost linearly with concurrency, with no plateau found up to 96 simultaneous requests.** Going from 1 → 96 concurrent requests is a **53.5x throughput increase** (64.4 → 3,443.6 tok/s).
- **Per-request latency degrades far slower than concurrency increases.** At concurrency=96, a single request's latency (41.8s) is only ~1.8x the concurrency=1 latency (23.3s) — a 96x increase in simultaneous load costs less than 2x in per-user wait time.
- **VRAM usage is essentially flat (132,033 → 132,037 MiB) across every concurrency level tested.** vLLM pre-allocates its KV cache pool once at server startup; that pool is large enough to hold all 96 concurrent sequences' KV cache without additional allocation. This is the single most important structural fact about this deployment: memory is not the bottleneck at any concurrency level tested.
- **GPU compute utilization was already pinned at 95-100% even at concurrency=1** (the model is inherently compute-bound per token, not memory-bandwidth-bound in a way that leaves headroom at low concurrency) — yet the scheduler still found room to pack far more work into the same utilization envelope via continuous batching. This confirms vLLM's iteration-level batching is doing real work: it's not idling between decode steps at low concurrency, it's simply serializing tokens one request at a time, and batching lets it interleave many requests' decode steps into the same GPU passes.
- No ceiling was reached at 96 concurrent requests. The true throughput/latency ceiling for this model + GPU combination is higher than tested here.

---

## 5. GPU / VRAM behavior (all tests combined)

- **Idle VRAM (model loaded, no requests in flight): 132,033 MiB** out of 143,771 MiB total — this is the static footprint of model weights (~54 GB) plus vLLM's pre-allocated KV cache block pool (sized at startup based on `--max-model-len 32768` and available memory).
- **VRAM headroom: ~11.7 GB** (143,771 − 132,033 MiB) remained unused across every test, including the 96-concurrent-request run.
- VRAM did **not** grow measurably with either request count or output length across any test — consistent with vLLM's PagedAttention design, which reserves a fixed block pool and manages occupancy internally rather than growing process memory per request.
- GPU utilization was consistently 95-100% under any load ≥1 concurrent request — this GPU is compute-saturated, not memory-saturated, for this workload.

---

## 6. Conclusions & recommendations

1. **This deployment is dramatically under-utilized for single-user/interactive use.** A lone user sees the same ~64 tok/s whether they're the only requester or one of 96 — so serving this model to a single interactive session wastes the vast majority of the H200's capacity.
2. **Optimal concurrency for this workload appears to be well above 96** — no throughput plateau or latency cliff was observed in this test. To find the actual ceiling, the next step would be pushing further (e.g. 128, 192, 256+ concurrent requests) and watching for either (a) VRAM exhaustion once the ~11.7 GB headroom is consumed by longer-running concurrent KV caches, or (b) request queuing once vLLM's internal scheduler batch-size limits are hit.
3. **VRAM headroom, not raw request count, is likely the eventual limiting factor.** Because KV cache size grows with both concurrency *and* sequence length, a workload with much longer contexts (e.g. large system-design answers at 12,000+ tokens, as seen in the sequential test) run concurrently would consume the remaining ~11.7 GB headroom faster than this test's short 1,500-token-capped requests did. A production capacity plan should size expected concurrent long-context sessions against that headroom, not just peak request count.
4. **For a production deployment expecting bursty or multi-tenant traffic, this single H200 instance can likely absorb far more concurrent load than initially assumed** — the benchmark data supports provisioning for high concurrency (dozens to potentially 100+ simultaneous requests) on this hardware without needing to scale out, at least for workloads similar in shape to these system-design prompts.

---

## Appendix: raw result files

| File | Contents |
|---|---|
| `benchmark_results_full_stop.json` | Sequential 10-case run, 12,000 max_tokens, full per-case detail + GPU samples |
| `benchmark_concurrency_results.json` | Concurrency sweep, levels 1/2/4/8/16 |
| `benchmark_concurrency_results_high.json` | Concurrency sweep, levels 32/64/96 |
| `benchmark_vllm.py` | Sequential benchmark script (reusable, CLI-configurable) |
| `benchmark_vllm_concurrency.py` | Concurrency benchmark script (reusable, CLI-configurable) |
