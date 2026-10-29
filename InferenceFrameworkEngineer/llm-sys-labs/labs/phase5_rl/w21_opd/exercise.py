# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 21: on-policy distillation (OPD).

Roadmap: "Implement OPD: sample from the student, score every token with a
teacher's log-probs, minimise per-token reverse KL. Compare cost and accuracy
against GRPO on the same task."

Thinking Machines (Oct 2025) frame it as RL whose reward is dense: every
sampled token gets a grade from the teacher instead of one bit per episode.

Two estimators of the per-token reverse KL, KL(pi_student || pi_teacher):
  sampled - only the teacher's log-prob of the token the student sampled is
            needed (what an API or a big remote teacher gives you):
              advantage_t = -(log pi_s(x_t) - log pi_t(x_t)),
            then a policy-gradient step  -advantage_t * log pi_s(x_t).
            This is the Tinker cookbook recipe.
  full    - with full teacher logits (small vocab, local teacher) compute the
            exact KL over the vocabulary at every student-visited prefix and
            backprop through it directly. Lower variance, more teacher I/O.

GRPO baseline uses the same student start, prompts and sample counts.

Measured on this CPU setup (tiny GPT, 2-digit addition, student starts at 21.5%,
30 steps x 128 samples, lr 5e-4):
    independent-init teacher (91%):  GRPO 42.0%  OPD-sampled 37.0%  OPD-full 44.0%
    same-lineage teacher (97%):      GRPO 42.0%  OPD-sampled 40.5%  OPD-full 45.5%
    at lr 2e-3, 60 steps:            GRPO  9.0%  OPD-sampled 13.0%  OPD-full 24.0%
So at toy scale OPD is comparable to GRPO, the exact-KL estimator is a little
better and much more robust to a high learning rate, and the sampled estimator
is noisier. That is NOT the large win Thinking Machines report at Qwen3 scale
(their AIME'24 result: 70% with ~1,800 GPU-hours vs ~17,920 for RL). Exercise:
find out why. Candidates: student capacity, answer length (3 tokens leaves
little room for dense per-token credit), teacher/student compatibility (the
April 2026 "Rethinking On-Policy Distillation" study reports this matters).
"""
from __future__ import annotations

import copy
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F

from common.rl_task import accuracy, completion_mask, make_problem, new_policy, reward, sample, sft, token_logprobs


def make_teacher(seed: int = 0, min_acc: float = 0.8) -> torch.nn.Module:
    """A well-trained teacher: same architecture, 800 SFT steps, independent init.

    SFT on this task has a sharp, seed-dependent phase transition (measured:
    seeds 0 and 7 reach 97% and 91% at 800 steps; seed 100 stays near 3%). A
    weak teacher makes OPD meaningless, so refuse to continue with one."""
    t = sft(new_policy(seed + 7), 800, seed=seed + 7).eval()
    acc = accuracy(t)
    if acc < min_acc:
        raise RuntimeError(f"teacher accuracy {acc:.1%} < {min_acc:.0%}; pick another seed")
    return t


def opd_loss_sampled(student, teacher, seqs, mask):
    """Policy gradient with per-token advantage = -(log p_s - log p_t) on sampled tokens."""
    # TODO: logp_s with grad; logp_t no grad; adv = -(logp_s.detach() - logp_t); loss = -(adv * logp_s * mask).sum() / mask.sum()
    raise NotImplementedError("TODO(w21_opd): logp_s with grad; logp_t no grad; adv = -(logp_s.detach() - logp_t); loss = -(adv * logp_s * mask).sum() / mask.sum()")


def opd_loss_full(student, teacher, seqs, mask, prompt_len: int = 6):
    """Exact reverse KL over the vocabulary at each student-visited completion prefix."""
    # TODO: student/teacher log_softmax at completion positions; kl = sum_v p_s * (log p_s - log p_t); masked mean
    raise NotImplementedError("TODO(w21_opd): student/teacher log_softmax at completion positions; kl = sum_v p_s * (log p_s - log p_t); masked mean")


def grpo_loss(student, ref, seqs, rewards, G, mask):
    r = rewards.view(-1, G)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).flatten()
    logp = token_logprobs(student, seqs)
    with torch.no_grad():
        lref = token_logprobs(ref, seqs)
    d = lref - logp
    per = -adv[:, None] * logp + 0.04 * (torch.exp(d) - d - 1)  # on-policy: ratio = 1
    return (per * mask).sum() / mask.sum()


def run(method: str, steps: int = 30, P: int = 16, G: int = 8, lr: float = 5e-4, seed: int = 0, teacher=None) -> dict:
    student = sft(new_policy(seed), 300, seed=seed)
    ref = copy.deepcopy(student).eval()
    teacher = teacher if teacher is not None else make_teacher(seed)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    rnd = random.Random(seed + 1)
    gen = torch.Generator().manual_seed(seed)
    out = {"method": method, "acc_start": accuracy(student), "teacher_tokens": 0}
    t0 = time.perf_counter()
    for _ in range(steps):
        prompts = [make_problem(rnd)[0] for _ in range(P)]
        seqs = sample(student, prompts, G, temperature=1.0, generator=gen)
        mask = completion_mask(seqs)
        if method == "grpo":
            rew = torch.tensor([reward(prompts[i // G], seqs[i, 6:].tolist()) for i in range(len(seqs))])
            loss = grpo_loss(student, ref, seqs, rew, G, mask)
        elif method == "opd_sampled":
            loss = opd_loss_sampled(student, teacher, seqs, mask)
            out["teacher_tokens"] += int(mask.sum())
        elif method == "opd_full":
            loss = opd_loss_full(student, teacher, seqs, mask)
            out["teacher_tokens"] += int(mask.sum())
        else:
            raise ValueError(method)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
    out["wall_s"] = time.perf_counter() - t0
    out["acc_end"] = accuracy(student)
    return out


def compare(steps: int = 30, seed: int = 0) -> list[dict]:
    teacher = make_teacher(seed)
    rows = [{"method": "teacher", "acc_end": accuracy(teacher)}]
    rows += [run(m, steps, seed=seed, teacher=teacher) for m in ("grpo", "opd_sampled", "opd_full")]
    return rows


if __name__ == "__main__":
    from common.bench import table

    print(table(compare(), ["method", "acc_start", "acc_end", "wall_s", "teacher_tokens"]))
