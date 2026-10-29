# Phase 1 (weeks 1-4): GPU and PyTorch foundations

Goal: read a profile and say whether a step is compute-, memory- or overhead-bound.

| Week | Lab | You implement | Runs on |
| --- | --- | --- | --- |
| 1 | `w01_roofline` | arithmetic intensity of GEMM / softmax / LayerNorm, ridge point, roofline time | CPU, GPU |
| 2 | `w02_kernels` | vector add, block reduction (warp shuffles), 32x32 shared-memory SGEMM in CUDA; the same three as tiled C++/OpenMP for CPU | CPU (C++), GPU (CUDA) |
| 3 | `w03_pytorch_internals` | RMSNorm `autograd.Function` with hand-derived backward, a `torch.library.custom_op` with fake kernel, saved-activation accounting, profiler top-ops, graph-break removal | CPU, GPU |
| 4 | `w04_gpt_trainer` | causal attention, warmup+cosine LR, decay groups, BF16 / checkpointing / `torch.compile` ablation with MFU | CPU (tiny), GPU (`--preset gpt2`, 124M) |

```bash
LAB=exercise pytest labs/phase1_foundations/w02_kernels     # your work
python labs/phase1_foundations/w02_kernels/exercise.py --bench
ncu --set full -k regex:sgemm python labs/phase1_foundations/w02_kernels/exercise.py --bench   # GPU
```

**Gate.** Tiled SGEMM at 4096^2 >= 70% of cuBLAS (GPU); GPT-2 124M trainer with a tokens/s + MFU table per optimisation; one-page roofline of your training step.

**Measured on the 2-core CPU used to build these labs** (reference solutions):
SGEMM twin reached 26-40% of MKL at n=512 (room to improve with register blocking).
Week 4 ablation over two runs: fp32 45-47K tok/s; BF16 autocast **17-20K tok/s (2.4-2.6x slower: no BF16 matrix units on that CPU)**; checkpointing 10-27% slower; `torch.compile` 18-25% faster. Explain each number in your write-up.

**GPU status.** `solution_kernels.cu` compiles with nvcc 13.2 for sm_90 against PyTorch 2.15 headers; not executed (no GPU in the build environment).
