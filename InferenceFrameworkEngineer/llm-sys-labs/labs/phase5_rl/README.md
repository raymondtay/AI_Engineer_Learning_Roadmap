# Phase 5 (weeks 18-21): RL training systems

Goal: own the RL loop as a systems problem: generation, training, weight sync, staleness.

Shared task (`common/rl_task.py`): 2-digit addition with an exact-match verifier, a byte-level tiny GPT, and an SFT warm start (300 steps -> 21.5% accuracy) so RL has room to improve.

| Week | Lab | You implement | Measured on CPU |
| --- | --- | --- | --- |
| 18 | `w18_grpo_sync` | group-normalised advantages, PPO-clipped token loss + k3 KL, the sync loop with a gen/train timeline | 21.5% -> 42-44.5% in 30-40 steps; generation 66% of wall time |
| 19 | `w19_async_rl` | versioned weight store, rollout thread, bounded queue, staleness filter, behaviour-policy ratio | async/sync wall time 0.94-1.17x on 2 cores (noisy); staleness bounded as set |
| 20 | `w20_agent_rl` | AST-whitelisted calculator sandbox, multi-turn rollout splicing env tokens as ids, action mask | tool use 34-54% -> 95-97%; reward roughly doubles |
| 21 | `w21_opd` | on-policy distillation: sampled per-token reverse-KL advantage and exact full-vocab reverse KL | see below |

**OPD result, reported as measured.** With a verified 91-97% teacher and the student at 21.5%, 30 steps: GRPO 42%, OPD-sampled 37-40.5%, OPD-full 44-45.5%. At a high LR (2e-3) OPD-full degraded least (24% vs GRPO 9%). This does **not** reproduce the large OPD advantage Thinking Machines report at Qwen3 scale; the lab's exercise is to find out why.

**Async RL on CPU.** The two threads share 2 cores, so the overlap gain is small and varies run to run. The mechanism (staleness, versioning, interruptible weight pulls) is what this lab verifies; measure the speedup on GPUs where rollout and training have separate devices.

**GPU.** `gpu/verl_grpo.sh` uses GRPO config keys verified against verl's GRPO docs (others marked `VERIFY`); not executed. For async and agentic RL at scale read slime's `examples/fully_async` and AReaL; for OPD the Tinker cookbook `recipes/distillation`.
