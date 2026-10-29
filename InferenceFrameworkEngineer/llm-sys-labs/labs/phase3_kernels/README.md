# Phase 3 (weeks 10-13): kernels in Triton, TileLang and CUDA

Goal: write the fused kernels FP8 training and fast inference depend on.

| Week | Lab | You implement | How it runs on CPU |
| --- | --- | --- | --- |
| 10 | `w10_triton_basics` | fused softmax, RMSNorm fwd + bwd (autograd), autotuned tiled matmul | Triton interpreter (`TRITON_INTERPRET=1`, set automatically) |
| 11 | `w11_flash_attention` | FlashAttention-2 forward (online softmax, LSE), backward (dK/dV and dQ kernels), causal + packed varlen | Triton interpreter; matches SDPA fwd and bwd |
| 12 | `w12_tilelang` | tiled GEMM, FP8 block-scaled GEMM with DeepSeek-style promotion, attention with online softmax | TileLang C backend (`target="c"`) for scalar tile programs |
| 13 | `w13_cuda_depth` | `mma.sync.m16n8k16` fragment layouts, bank-conflict counting, XOR swizzle, cp.async pipeline model | exact models of the hardware rules; real `mma.sync` check on GPU |

```bash
LAB=exercise pytest labs/phase3_kernels/w11_flash_attention
python labs/phase3_kernels/w11_flash_attention/solution.py    # GPU: speed vs flash_attn / SDPA
```

**Gate (GPU).** Triton FlashAttention fwd+bwd within 1.2x of FlashAttention-2 at 4K; TileLang FP8 block-scaled GEMM vs cuBLAS FP8; fused RMSNorm in your Phase 2 model with step-time change.

**What the CPU path cannot tell you.** Speed. The interpreter executes kernels with NumPy, so it verifies indexing, masking and maths, not performance. TileLang's GPU primitives (`alloc_fragment`, `T.gemm`, `T.Pipelined`) do not lower to the C backend; each kernel has a `*_cpu` scalar version (verified) and a `*_gpu` version (GPU-only).

**Exercise caveat.** Inside `@triton.jit` kernels the generated `raise NotImplementedError` works under the interpreter; on a GPU Triton reports it as a compilation error pointing at the TODO line.

**GPU status.** `mma_check.cu` compiles with nvcc 13.2 for sm_90 (inline PTX assembled by ptxas); not executed. For MLA decode and a T.gemm FlashAttention, start from the official TileLang examples.
