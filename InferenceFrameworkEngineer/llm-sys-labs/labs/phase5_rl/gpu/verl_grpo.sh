#!/usr/bin/env bash
# Week 18 (GPU): GRPO with verl - FSDP2 training + vLLM or SGLang rollouts.
#
# STATUS: not executed by the author (no GPU in the build environment).
# Verified against verl's GRPO doc page (v0.5.x): algorithm.adv_estimator=grpo,
# actor_rollout_ref.rollout.n, data.train_batch_size,
# actor_rollout_ref.actor.use_kl_loss / kl_loss_coef / kl_loss_type (low_var_kl = k3),
# actor_rollout_ref.actor.loss_agg_mode, and the example script
# examples/grpo_trainer/run_qwen3-8b.sh. Keys marked VERIFY come from verl's
# quick-start conventions; confirm them against your pinned verl version.
#
# Easiest path: run the maintained example, then override what the lab needs:
#   cd verl && bash examples/grpo_trainer/run_qwen3-8b.sh
#
# Lab version (1.5B model, GSM8K preprocessed with examples/data_preprocess/gsm8k.py):
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
ROLLOUT=${ROLLOUT:-vllm}   # or sglang
DATA=${DATA:-$HOME/data/gsm8k}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA/train.parquet" \
  data.val_files="$DATA/test.parquet" \
  data.train_batch_size=256 \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.rollout.name="$ROLLOUT" \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.strategy=fsdp2 \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  2>&1 | tee verl_grpo.log
# VERIFY: actor_rollout_ref.actor.strategy=fsdp2, trainer.* keys.
#
# Timeline gate: verl's FAQ documents generating a Ray timeline
# ("How to generate ray timeline to analyse performance of a training job?").
# Record generation vs training vs resharding seconds per step, the same split
# solution.py prints for the CPU loop.
