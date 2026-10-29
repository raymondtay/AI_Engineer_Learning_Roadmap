# Phase 2 (weeks 5-9): distributed training, long context, MoE, FP8

Goal: hand-build each parallelism once, then run the production version in Megatron-Core and explain its numbers.

All distributed labs launch N processes with `common/dist.py`: **gloo on CPU, NCCL on GPU, same code**. Every result is checked against a single-process reference.

| Week | Lab | You implement | Verified on CPU |
| --- | --- | --- | --- |
| 5 | `w05_ddp_zero` | nccl-tests bus-bandwidth formula, bucketed DDP all-reduce, ZeRO-2 (reduce-scatter grads, sharded Adam, all-gather); FSDP2 `fully_shard` comparison | DDP and ZeRO match full-batch SGD/Adam to < 1e-5; FSDP2 runs on gloo |
| 6 | `w06_tp_pp` | Megatron f/g operators, column/row-parallel MLP and attention, 1F1B schedule, real 1F1B pipeline over send/recv | TP=2,4 and PP=2,4 match single-device grads exactly |
| 7 | `w07_context_parallel` | Korthikanti activation-memory budget, ring attention with online softmax, Ulysses all-to-all | CP=2,4 match full causal attention (< 1e-5) |
| 8 | `w08_moe_ep` | top-k router, Switch aux loss, DeepSeek aux-loss-free bias balancing, expert parallel with uneven all-to-all | EP=2,4 exact; load max/mean 3.14 -> 1.09 |
| 9 | `w09_fp8` | E4M3/E5M2 quantisation, block-wise scales, delayed scaling, FP8 linear (E4M3 fwd, E5M2 grads) | outlier 1e5: per-tensor scale zeroes 17.7% of values, 1x128 blocks 0.15% |

```bash
LAB=exercise pytest labs/phase2_distributed/w06_tp_pp
torchrun --nproc-per-node 8 ...   # on a real node; see each lab's docstring
bash labs/phase2_distributed/gpu/megatron_gate.sh dense   # GPU gate (Megatron-Core)
python labs/phase2_distributed/w09_fp8/gpu_te_fp8.py      # GPU gate (Transformer Engine FP8)
```

**Gate (GPU).** From-scratch 2D TP x DP matching single-GPU loss; 1-3B model in Megatron-Core at 40%+ MFU on 8 GPUs; FP8 vs BF16 gap < 0.25%; MoE trace showing all-to-all overlap.

**Honest note on FP8.** The CPU emulation shows FP8 trailing fp32 by ~6% of mean loss on a 150-step tiny run, catching up by the end. That is a toy-scale number; the 0.25% gate is for the multi-thousand-step Transformer Engine run.

**GPU status.** `megatron_gate.sh` and `gpu_te_fp8.py` were not executed. Flags in the Megatron script were checked against NVIDIA's Megatron-Core docs where shown there; others are marked `VERIFY`.
