# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 18: synchronous GRPO baseline, from scratch.

Roadmap: "Run GRPO on a 1.5-7B model in verl with FSDP2 + vLLM or SGLang
rollouts. Profile the timeline: generation vs training vs resharding."

CPU: the full GRPO loop on the shared verifiable task (2-digit addition) with
the tiny GPT, warm-started by SFT. Same algorithm verl runs, minus the
distributed engines. GPU: gpu_verl_grpo.sh runs the real thing.

GRPO (Shao et al. 2024, DeepSeekMath):
  1. for each prompt sample G completions from the current policy
  2. reward each; advantage = (r - mean_group) / (std_group + eps)  (no critic)
  3. PPO-clipped surrogate per token with that advantage, plus a KL penalty
     to a frozen reference policy (k3 estimator: exp(d) - d - 1, d = ref - new)

The timeline (`stats["gen_s"]`, `stats["train_s"]`) shows the synchronous
system's problem: the trainer idles while generation runs, and vice versa.
Week 19 removes that.
"""
from __future__ import annotations

import copy
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from common.rl_task import accuracy, completion_mask, make_problem, new_policy, reward, sample, sft, token_logprobs


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """rewards: (P, G). Returns (P, G) normalised within each group."""
    # TODO: (r - mean over G) / (std over G + eps)
    raise NotImplementedError("TODO(w18_grpo_sync): (r - mean over G) / (std over G + eps)")


def grpo_loss(logp: torch.Tensor, logp_old: torch.Tensor, logp_ref: torch.Tensor, adv: torch.Tensor,
              mask: torch.Tensor, clip: float = 0.2, beta: float = 0.04) -> torch.Tensor:
    """Token-level GRPO loss. logp*: (N, T); adv: (N,); mask: (N, T)."""
    # TODO: ratio = exp(logp - logp_old); surrogate = -min(ratio*A, clip(ratio)*A); kl = exp(ref-logp) - (ref-logp) - 1; mean over masked tokens
    raise NotImplementedError("TODO(w18_grpo_sync): ratio = exp(logp - logp_old); surrogate = -min(ratio*A, clip(ratio)*A); kl = exp(ref-logp) - (ref-logp) - 1; mean over masked tokens")


def grpo_step(policy, ref, opt, prompts: list[str], G: int, gen: torch.Generator, stats: dict, inner_epochs: int = 1):
    t0 = time.perf_counter()
    # TODO: sample G per prompt; rewards (P, G); advantages; old and ref logprobs under no_grad
    raise NotImplementedError("TODO(w18_grpo_sync): sample G per prompt; rewards (P, G); advantages; old and ref logprobs under no_grad")
    t1 = time.perf_counter()
    for _ in range(inner_epochs):
        loss = grpo_loss(token_logprobs(policy, seqs), logp_old, logp_ref, adv, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    t2 = time.perf_counter()
    stats["gen_s"] += t1 - t0
    stats["train_s"] += t2 - t1
    stats["mean_reward"].append(rew.mean().item())
    return loss.item()


def train(steps: int = 40, prompts_per_step: int = 16, G: int = 8, sft_steps: int = 300, lr: float = 5e-4, seed: int = 0):
    policy = sft(new_policy(seed), sft_steps, seed=seed)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    rnd = random.Random(seed + 1)
    gen = torch.Generator().manual_seed(seed)
    stats = {"gen_s": 0.0, "train_s": 0.0, "mean_reward": [], "acc_start": accuracy(policy)}
    for _ in range(steps):
        prompts = [make_problem(rnd)[0] for _ in range(prompts_per_step)]
        grpo_step(policy, ref, opt, prompts, G, gen, stats)
    stats["acc_end"] = accuracy(policy)
    return policy, stats


if __name__ == "__main__":
    _, s = train()
    tot = s["gen_s"] + s["train_s"]
    print(f"accuracy {s['acc_start']:.1%} -> {s['acc_end']:.1%}")
    print(f"timeline: generation {s['gen_s']:.1f}s ({s['gen_s']/tot:.0%}), training {s['train_s']:.1f}s ({s['train_s']/tot:.0%})")
