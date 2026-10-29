# Gate results (solution)

2026-10-29 · CPU x2 · torch 2.15.0+cu132 · Linux-6.18.44-fc-v185-x86_64-with-glibc2.39

| Gate | Result |
| --- | --- |
| W01 roofline | bandwidth 24.1 GB/s, matmul 0.245 TFLOP/s, ridge 10.2 FLOP/B |
| W02 CUDA/C++ kernels | SGEMM 26% of library at n=512 (GPU gate: >= 70% of cuBLAS at 4096) |
| W04 GPT trainer | baseline fp32: 45203 tok/s, MFU 16.3%; + bf16 autocast: 17345 tok/s, MFU 6.3%; + activation checkpointing: 32969 tok/s, MFU 11.9%; + torch.compile: 53233 tok/s, MFU 19.2% |
| W05 DDP/ZeRO | ZeRO-2 vs Adam max |diff| 3.0e-07; Adam state per rank 27168 B |
| W06 TP/PP | TP=2 out diff 2.2e-07, dx diff 2.4e-07; 1F1B bubble p=4 m=16: 15.8% |
| W07 context parallel | CP=4 ring diff 2.4e-07, Ulysses diff 0.0e+00 |
| W08 MoE/EP | EP=4 diff 0.0e+00; load max/mean 3.14 -> 1.09 (aux-free) |
| W09 FP8 | FP8 vs fp32 mean-loss gap 6.5% (CPU emulation; GPU gate < 0.25% with TE) |
| W11 FlashAttention | Triton FA correctness diff 4.8e-07 (speed gate needs GPU) |
| W12 TileLang | TileLang block-scaled FP8 GEMM diff 1.6e-05 |
| W14 engine | engine 1228 tok/s, 2.4x naive (GPU gate: within 2x of vLLM) |
| W15 prefix cache | prefill saved: multi-turn 78%, rag 87%; TTFT 26.3 -> 21.1 ms |
| W16 tiered KV | multi-turn TTFT 26.2 -> 16.7 ms; no-reuse overhead +9% |
| W17 PD + router | P50 TTFT rr 23.4 vs cache-aware 17.0 ms; P99 95 vs 67 ms; reuse 54% vs 71% |
| W18 GRPO | GRPO acc 22% -> 42%; generation 67% of time |
| W19 async RL | async 0.94x vs sync; acc sync 37% / async 48%; max staleness 2 |
| W20 agent RL | tool use 34% -> 95%; reward 0.08 -> 0.33 |
| W21 OPD | grpo 22% -> 42%; opd_full 22% -> 44% |
| W22 VLM | caption acc 95%; saved act. 8.9 MB -> 6.2 MB (ViT recompute) |
| W23 data loading | stall 25% -> 5% with 2 workers; token efficiency 66% padded -> 98% packed |
