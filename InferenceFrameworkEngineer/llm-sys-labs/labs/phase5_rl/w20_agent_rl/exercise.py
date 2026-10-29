# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 20: agent RL - multi-turn tool calls, sandboxed execution, verifier reward.

Roadmap: "Multi-turn tool-calling rollouts with a sandbox (code execution) and
verifier rewards. Handle long-tail trajectories and tokenisation across turns."

Task: 2-digit multiplication ("37*48="), which the tiny model cannot do in its
head but can solve by calling a calculator. Byte-level protocol:

    policy:  37*48=<                 '<' asks the environment to run the tool
    env:     >1776;                  tool output, appended as token ids
    policy:  1776\n                  final answer, verified for reward

Systems points this lab makes concrete:
  * Environment tokens enter the context but are NOT the policy's actions:
    the loss mask must exclude them (otherwise the policy is trained to
    "predict" the tool's output, and the gradient is wrong).
  * Keep token ids end to end. Decoding to text and re-tokenising each turn can
    produce a different token sequence than the one the policy sampled, so the
    log-probs you train on no longer match the actions taken (verl's
    "When Reasoning Models Break Tokenization" post covers this failure).
  * The sandbox evaluates a whitelisted grammar only - never eval() model output.
  * Trajectories have different lengths (tool vs no tool): batches are ragged
    and padded; a per-step token budget bounds the long tail.
"""
from __future__ import annotations

import ast
import copy
import operator
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F

from common.rl_task import dec, enc
from common.tinygpt import GPT, GPTConfig

CALL, OPEN, CLOSE, NL, PAD = ord("<"), ord(">"), ord(";"), ord("\n"), 0
MAX_TOKENS = 12  # policy-generated tokens per trajectory (long-tail cap)


# ------------------------------------------------------------------ sandbox

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def run_tool(expr: str) -> str:
    """Evaluate +, -, * on integers. Anything else -> 'ERR'."""
    # TODO: parse with ast in 'eval' mode; recursively allow only BinOp with whitelisted ops over int Constants; cap operand size; return str(result) or 'ERR'
    raise NotImplementedError("TODO(w20_agent_rl): parse with ast in 'eval' mode; recursively allow only BinOp with whitelisted ops over int Constants; cap operand size; return str(result) or 'ERR'")


# ------------------------------------------------------------------ task


def problem(rnd: random.Random) -> tuple[str, int]:
    a, b = rnd.randint(10, 99), rnd.randint(10, 99)
    return f"{a}*{b}=", a * b


def verify(prompt: str, policy_answer: list[int]) -> float:
    a, b = int(prompt[:2]), int(prompt[3:5])
    return 1.0 if dec(policy_answer).split("\n")[0] == str(a * b) else 0.0


def new_agent(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(seq_len=32, n_layer=2, n_head=4, d_model=64))


def sft_format(model: GPT, steps: int = 300, p_tool: float = 0.3, seed: int = 0) -> GPT:
    """Warm start: 30% of demos use the tool, 70% answer directly (and the model
    cannot multiply, so direct answers are mostly wrong at test time)."""
    rnd = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        rows = []
        for _ in range(64):
            p, ans = problem(rnd)
            body = f"<>{ans};{ans}\n" if rnd.random() < p_tool else f"{ans}\n"
            rows.append(enc(p + body))
        L = max(map(len, rows))
        x = torch.tensor([r + [PAD] * (L - len(r)) for r in rows])
        m = torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in rows])
        m[:, :6] = 0
        _, loss = model(x[:, :-1], x[:, 1:], loss_mask=m[:, 1:].bool())
        loss.backward()
        opt.step()
        opt.zero_grad()
    return model


# ------------------------------------------------------------------ rollout


@torch.no_grad()
def rollout(model: GPT, prompt: str, gen: torch.Generator, temperature: float = 1.0):
    """One multi-turn trajectory. Returns (ids, action_mask, used_tool, answer_ids)."""
    ids = enc(prompt)
    mask = [0] * len(ids)
    used_tool, answer_start, n_gen = False, None, 0
    # TODO: sample token by token; on CALL (first time) append it (mask 1) then env tokens OPEN+result+CLOSE (mask 0); on NL stop; cap at MAX_TOKENS generated tokens; track where the final answer starts
    raise NotImplementedError("TODO(w20_agent_rl): sample token by token; on CALL (first time) append it (mask 1) then env tokens OPEN+result+CLOSE (mask 0); on NL stop; cap at MAX_TOKENS generated tokens; track where the final answer starts")
    return ids, mask, used_tool, ids[answer_start:]


def pad(rows, value):
    L = max(map(len, rows))
    return torch.tensor([r + [value] * (L - len(r)) for r in rows])


def seq_logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    logits, _ = model(ids[:, :-1])
    return F.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None])[..., 0]


def train(steps: int = 25, P: int = 8, G: int = 8, lr: float = 1e-3, seed: int = 0) -> dict:
    policy = sft_format(new_agent(seed), seed=seed)
    ref = copy.deepcopy(policy).eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    rnd = random.Random(seed + 7)
    gen = torch.Generator().manual_seed(seed)
    stats = {"reward": [], "tool_rate": []}
    for _ in range(steps):
        trajs, rewards = [], []
        for _ in range(P):
            prompt, _ = problem(rnd)
            group = [rollout(policy, prompt, gen) for _ in range(G)]
            trajs += group
            rewards.append([verify(prompt, g[3]) for g in group])
        r = torch.tensor(rewards)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).flatten()
        ids = pad([t[0] for t in trajs], PAD)
        amask = pad([t[1] for t in trajs], 0)[:, 1:].float()  # action mask aligned with next-token targets
        with torch.no_grad():
            old, refl = seq_logprobs(policy, ids), seq_logprobs(ref, ids)
        new = seq_logprobs(policy, ids)
        ratio = torch.exp(new - old)
        surr = -torch.minimum(ratio * adv[:, None], ratio.clamp(0.8, 1.2) * adv[:, None])
        d = refl - new
        loss = ((surr + 0.04 * (torch.exp(d) - d - 1)) * amask).sum() / amask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        stats["reward"].append(r.mean().item())
        stats["tool_rate"].append(sum(t[2] for t in trajs) / len(trajs))
    return stats


if __name__ == "__main__":
    s = train()
    k = 5
    print(f"reward    first {k} steps {sum(s['reward'][:k])/k:.2f} -> last {k} {sum(s['reward'][-k:])/k:.2f}")
    print(f"tool use  first {k} steps {sum(s['tool_rate'][:k])/k:.2f} -> last {k} {sum(s['tool_rate'][-k:])/k:.2f}")
