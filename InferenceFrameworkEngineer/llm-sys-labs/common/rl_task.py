"""Shared verifiable task and policy utilities for Phase 5 (RL systems).

Task: two-digit addition, "a+b=" -> "c\n". Verifiable reward = exact match.
Small enough to train on CPU in seconds, hard enough that a warm-started
policy is far from perfect, so RL has something to fix.

Everything is byte-level (vocab 256) and uses the shared tiny GPT.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from common.tinygpt import GPT, GPTConfig

NL = ord("\n")
MAX_NEW = 4  # up to 3 answer digits + newline


def make_problem(rnd: random.Random, lo: int = 0, hi: int = 50) -> tuple[str, str]:
    a, b = rnd.randint(lo, hi), rnd.randint(lo, hi)
    return f"{a:02d}+{b:02d}=", f"{a + b}\n"


def enc(s: str) -> list[int]:
    return list(s.encode())


def dec(t) -> str:
    return bytes(int(x) for x in t).decode(errors="replace")


def reward(prompt: str, completion: list[int]) -> float:
    """1.0 if the completion's text up to the first newline is the right sum."""
    a, b = int(prompt[0:2]), int(prompt[3:5])
    text = dec(completion).split("\n")[0]
    return 1.0 if text == str(a + b) else 0.0


def new_policy(seed: int = 0, d_model: int = 64, n_layer: int = 2) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(seq_len=16, n_layer=n_layer, n_head=4, d_model=d_model))


def sft(model: GPT, steps: int, seed: int = 0, lr: float = 3e-3, batch: int = 64) -> GPT:
    """Supervised warm start on correct examples (loss on answer tokens only)."""
    rnd = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        rows = [make_problem(rnd) for _ in range(batch)]
        seqs = torch.tensor([enc(p + a.ljust(MAX_NEW, "\n")) for p, a in rows])
        x, y = seqs[:, :-1], seqs[:, 1:]
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:, 5:] = True  # answer positions only (prompt is 6 bytes: "aa+bb=")
        _, loss = model(x, y, loss_mask=mask)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return model


@torch.no_grad()
def sample(model: GPT, prompts: list[str], n: int, temperature: float = 1.0, generator=None, max_new: int = MAX_NEW):
    """n completions per prompt. Returns (seqs (P*n, 6+max_new), completion ids list)."""
    base = torch.tensor([enc(p) for p in prompts]).repeat_interleave(n, 0)
    seq = base
    for _ in range(max_new):
        logits, _ = model(seq)
        probs = F.softmax(logits[:, -1].float() / max(temperature, 1e-5), -1)
        nxt = torch.multinomial(probs, 1, generator=generator) if temperature > 0 else probs.argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
    return seq


def token_logprobs(model: GPT, seqs: torch.Tensor, prompt_len: int = 6) -> torch.Tensor:
    """log pi(token_t | prefix) for every completion token. Shape (N, T_completion)."""
    logits, _ = model(seqs[:, :-1])
    lp = F.log_softmax(logits.float(), -1)
    tgt = seqs[:, 1:]
    out = lp.gather(-1, tgt[..., None])[..., 0]
    return out[:, prompt_len - 1 :]


def completion_mask(seqs: torch.Tensor, prompt_len: int = 6) -> torch.Tensor:
    """1 for completion tokens up to and including the first newline."""
    comp = seqs[:, prompt_len:]
    is_nl = (comp == NL).int()
    after = torch.cumsum(is_nl, 1) - is_nl  # newlines strictly before this position
    return (after == 0).float()


@torch.no_grad()
def accuracy(model: GPT, n: int = 200, seed: int = 123) -> float:
    rnd = random.Random(seed)
    probs = [make_problem(rnd)[0] for _ in range(n)]
    seqs = sample(model, probs, 1, temperature=0)
    return sum(reward(p, s[6:].tolist()) for p, s in zip(probs, seqs)) / n
