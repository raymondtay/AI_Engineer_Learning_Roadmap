# Phase 4 (weeks 14-17): inference systems, KV tiering, load balancing

Goal: build a small serving engine, then extend it with prefix reuse, a tiered KV store and a cache-aware router.

| Week | Lab | You implement | Measured on CPU (reference solution) |
| --- | --- | --- | --- |
| 14 | `w14_mini_engine` | block allocator, paged KV slots, continuous-batching scheduler with chunked prefill and recompute preemption, ragged forward | outputs identical to no-cache greedy under preemption; 2.4-3.4x naive throughput |
| 15 | `w15_prefix_cache` | vLLM-style chained block hashes, SGLang-style radix tree with edge splitting and LRU | prefill saved: multi-turn 78%, RAG 87%; TTFT 0.53x on multi-turn, +4-17% on no-reuse |
| 16 | `w16_tiered_kv` | device/host/disk tiers, LRU demotion, promotion on hit, async disk prefetch, KV service with exact reuse | multi-turn TTFT 26 -> 17 ms; no-reuse overhead +9% to +28% (disk write-through) |
| 17 | `w17_pd_router` | prefill/decode split over send/recv, KV bytes per token, round-robin / least-loaded / cache-aware routing | PD output == monolithic; cache-aware P50 TTFT -40% |

**Bugs worth reading in `w14`.** Two real scheduler bugs were found by the tests while building this lab and are documented in the code: a failed chunk allocation leaking half-allocated blocks, and preempted sequences jumping ahead of a partially prefilled head, which deadlocked the pool. `test_scheduler_stress` exists because of them.

**Routing result depends on the cost regime.** With default costs the cache-aware router improves both P50 and P99; with costs calibrated on this CPU (decode 5.5 ms/token vs prefill 0.14 ms/token) it improves P50 by 40% but leaves P99 flat, because the tail is queueing behind decode. Report which regime your GPU is in.

**GPU work.** Run the same three traces from `w15_prefix_cache` against vLLM (prefix caching, LMCache connector) and SGLang (RadixAttention, router). Mooncake / NIXL replace the send/recv in `w17`. No GPU scripts are included for these because the server CLIs change quickly; use each project's current docs and record versions.
