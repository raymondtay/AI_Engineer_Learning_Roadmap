"""Week 8: Mixture-of-Experts and expert parallel.

Roadmap: "Top-k router, aux-loss vs aux-loss-free balancing, all-to-all
dispatch. Profile comms/compute overlap in Megatron-Core."

Parts:
  1. Router              - softmax gate, top-k, renormalised weights
  2. switch_aux_loss     - Switch Transformer load-balancing loss: E * sum_i f_i * P_i
  3. AuxFreeBalancer     - DeepSeek-V3's bias-based balancing: a per-expert bias is
                           added to scores for *selection only* and nudged by
                           +/- gamma depending on whether the expert is under/over-loaded
  4. moe_forward_local   - reference single-process MoE layer
  5. ep_forward          - expert parallel: experts sharded across ranks, tokens
                           dispatched and combined with two uneven all-to-alls
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    def __init__(self, d: int, n_experts: int, k: int):
        super().__init__()
        self.gate = nn.Linear(d, n_experts, bias=False)
        self.k = k

    def forward(self, x, bias: torch.Tensor | None = None):
        """x: (N, d). Returns (idx (N,k), weights (N,k), probs (N,E))."""
        # >>> SOLUTION: probs = softmax(gate(x)); select top-k on probs (+bias if given); weights = probs at idx, renormalised
        probs = F.softmax(self.gate(x).float(), dim=-1)
        sel = probs if bias is None else probs + bias
        idx = sel.topk(self.k, dim=-1).indices
        w = probs.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)
        return idx, w, probs
        # <<< SOLUTION


def switch_aux_loss(probs: torch.Tensor, idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    """f_i = share of the N*k assignments that went to expert i; P_i = mean router prob."""
    # >>> SOLUTION: counts via one_hot(idx).sum over (N, k) divided by idx.numel(); P = probs.mean(0); return E * (f*P).sum()
    f = F.one_hot(idx, n_experts).float().sum((0, 1)) / idx.numel()
    P = probs.mean(0)
    return n_experts * (f * P).sum()
    # <<< SOLUTION


class AuxFreeBalancer:
    def __init__(self, n_experts: int, gamma: float = 1e-3):
        self.bias = torch.zeros(n_experts)
        self.gamma = gamma

    def update(self, idx: torch.Tensor):
        # >>> SOLUTION: load = tokens per expert; bias -= gamma * sign(load - mean load)
        load = torch.bincount(idx.flatten(), minlength=self.bias.numel()).float()
        self.bias -= self.gamma * torch.sign(load - load.mean())
        # <<< SOLUTION


def imbalance(idx: torch.Tensor, n_experts: int) -> float:
    load = torch.bincount(idx.flatten(), minlength=n_experts).float()
    return (load.max() / load.mean()).item()


class Expert(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.up, self.down = nn.Linear(d, hidden, bias=False), nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


def build(d=16, n_experts=8, k=2, hidden=32, seed=0):
    torch.manual_seed(seed)
    return Router(d, n_experts, k), nn.ModuleList(Expert(d, hidden) for _ in range(n_experts))


def moe_forward_local(x, router, experts):
    idx, w, _ = router(x)
    out = torch.zeros_like(x)
    # >>> SOLUTION: for each expert e, find (token, slot) pairs routed to e, run the expert once on those tokens, scatter-add weighted outputs
    for e, expert in enumerate(experts):
        tok, slot = (idx == e).nonzero(as_tuple=True)
        if tok.numel():
            out.index_add_(0, tok, expert(x[tok]) * w[tok, slot, None])
    # <<< SOLUTION
    return out


def ep_forward(x, router, experts, rank: int, world: int):
    """x: this rank's tokens (N, d). `experts` is the full list for weight
    sharing in the lab; this rank only *runs* experts [rank*L, (rank+1)*L)."""
    E = len(experts)
    L = E // world
    idx, w, _ = router(x)
    N, k = idx.shape
    flat_e = idx.flatten()                      # (N*k,) destination expert per assignment
    flat_tok = torch.arange(N).repeat_interleave(k)
    dest = flat_e // L                          # destination rank
    order = torch.argsort(dest, stable=True)
    send_x = x[flat_tok[order]]
    send_e = flat_e[order]
    send_counts = torch.bincount(dest, minlength=world)

    # >>> SOLUTION: exchange counts (all_to_all_single), then tokens and expert ids with uneven splits; run local experts; send results back the same way
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    sc, rc = send_counts.tolist(), recv_counts.tolist()
    recv_x = x.new_empty(sum(rc), x.shape[1])
    dist.all_to_all_single(recv_x, send_x.contiguous(), output_split_sizes=rc, input_split_sizes=sc)
    recv_e = send_e.new_empty(sum(rc))
    dist.all_to_all_single(recv_e, send_e.contiguous(), output_split_sizes=rc, input_split_sizes=sc)
    y = torch.empty_like(recv_x)
    for le in range(L):
        sel = recv_e == rank * L + le
        if sel.any():
            y[sel] = experts[rank * L + le](recv_x[sel])
    back = x.new_empty(send_x.shape)
    dist.all_to_all_single(back, y, output_split_sizes=sc, input_split_sizes=rc)
    # <<< SOLUTION

    out = torch.zeros_like(x)
    weights = w.flatten()[order]
    out.index_add_(0, flat_tok[order], back * weights[:, None])
    return out


def ep_check(rank: int, world: int, n_tokens: int = 24):
    router, experts = build()
    g = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(n_tokens, 16, generator=g)
    with torch.no_grad():
        ref = moe_forward_local(x, router, experts)
        out = ep_forward(x, router, experts, rank, world)
    return (out - ref).abs().max().item()


def balance_demo(steps: int = 300, n_experts: int = 8, gamma: float = 1e-2):
    """Skewed router: does aux-loss-free biasing flatten the load?"""
    torch.manual_seed(0)
    router = Router(16, n_experts, 2)
    with torch.no_grad():
        router.gate.weight[0] += 0.1  # with positive-mean inputs, expert 0 wins most tokens
    bal = AuxFreeBalancer(n_experts, gamma)
    x = torch.randn(512, 16) + 1.0
    first = None
    for _ in range(steps):
        with torch.no_grad():
            idx, _, _ = router(x, bal.bias)
        first = first or imbalance(idx, n_experts)
        bal.update(idx)
    return first, imbalance(idx, n_experts)


if __name__ == "__main__":
    from common import dist as ldist

    print("EP vs local, max |diff| per rank:", ldist.run(ep_check, 4))
    before, after = balance_demo()
    print(f"max/mean expert load: before {before:.2f} -> after aux-free biasing {after:.2f}")
