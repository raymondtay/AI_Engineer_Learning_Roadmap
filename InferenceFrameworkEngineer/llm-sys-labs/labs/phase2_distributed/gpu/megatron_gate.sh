#!/usr/bin/env bash
# Phase 2 gate (GPU only): a 1-3B dense model in Megatron-Core on one 8-GPU
# node at 40%+ MFU, then the same with context parallel, MoE and FP8.
#
# STATUS: not executed by the author (no GPU in the build environment).
# Verified against NVIDIA's Megatron-Core docs (Training Examples and Tokenizers
# pages): torchrun + pretrain_gpt.py, the model-shape / batch / LR / --bf16 /
# --log-interval flags, and `--tokenizer-type NullTokenizer --vocab-size N
# --mock-data` for benchmarking. Flags marked VERIFY were not on those pages:
# confirm with `python pretrain_gpt.py --help | grep -e <flag>` for your
# pinned Megatron-LM version before relying on them.
#
# Usage (inside nvcr.io/nvidia/pytorch container, from the Megatron-LM checkout):
#   bash megatron_gate.sh dense      # 1.3B dense, TP=1, PP=1, DP=8
#   bash megatron_gate.sh tp2pp2     # same model, TP=2 PP=2 DP=2: compare MFU
#   bash megatron_gate.sh cp         # 32K context with context parallel
#   bash megatron_gate.sh moe        # small MoE with expert parallel
#   bash megatron_gate.sh fp8        # dense + FP8 (Hopper/Ada/Blackwell)
set -euo pipefail
MODE=${1:-dense}
NGPU=${NGPU:-8}

COMMON=(
  --num-layers 24 --hidden-size 2048 --num-attention-heads 16
  --micro-batch-size 4 --global-batch-size 256
  --train-iters 200 --lr 3.0e-4 --min-lr 3.0e-5 --lr-decay-style cosine --lr-warmup-iters 20
  --weight-decay 0.1 --clip-grad 1.0
  --tokenizer-type NullTokenizer --vocab-size 50304 --mock-data
  --log-interval 10
  --bf16
)
SEQ=(--seq-length 2048 --max-position-embeddings 2048)

case "$MODE" in
  dense)  PAR=(--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1) ;;
  tp2pp2) PAR=(--tensor-model-parallel-size 2 --pipeline-model-parallel-size 2) ;;
  cp)     PAR=(--tensor-model-parallel-size 2 --context-parallel-size 4)          # VERIFY --context-parallel-size
          SEQ=(--seq-length 32768 --max-position-embeddings 32768)
          COMMON+=(--recompute-granularity selective) ;;                          # VERIFY
  moe)    PAR=(--tensor-model-parallel-size 1 --expert-model-parallel-size 8      # VERIFY
               --num-experts 8 --moe-router-topk 2) ;;                            # VERIFY
  fp8)    PAR=(--tensor-model-parallel-size 1)
          COMMON+=(--fp8-format hybrid) ;;  # VERIFY: older docs list --fp8-hybrid
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

torchrun --nproc_per_node="$NGPU" pretrain_gpt.py "${COMMON[@]}" "${SEQ[@]}" "${PAR[@]}" 2>&1 | tee "megatron_${MODE}.log"

# MFU from the log: Megatron prints elapsed time per iteration. Compute
#   tokens/s = global_batch * seq_len / iter_time
#   MFU      = tokens/s * (6*N + 12*L*h*s) / (NGPU * peak)   (see common/flops.py)
# and write it in your gate table next to the trace you took with nsys.
