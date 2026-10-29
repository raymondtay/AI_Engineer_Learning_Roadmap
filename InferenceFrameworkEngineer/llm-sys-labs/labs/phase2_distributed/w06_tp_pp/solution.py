"""Week 6: tensor parallel and pipeline parallel by hand.

Roadmap: "Hand-write Megatron-style column/row parallel MLP and attention.
Implement 1F1B pipeline schedule; measure the bubble."

Tensor parallel (Megatron-LM, Shoeybi et al. 2019):
  MLP:  Y = GeLU(X A) B.  Split A by columns, B by rows. Each rank computes a
        slice of the hidden layer with no communication; one all-reduce sums
        the partial outputs. Backward needs the mirror all-reduce on dX.
  The two conjugate ops are called f and g in the paper:
        f: identity forward, all-reduce backward   (enters the TP region)
        g: all-reduce forward, identity backward   (leaves the TP region)
  Attention: split heads across ranks; QKV is column-parallel, out-proj is
        row-parallel. Same f/g pattern.

Pipeline parallel: 1F1B (PipeDream-Flush / Megatron) schedule. With p stages
and m micro-batches the idle fraction ("bubble") is (p-1)/(m+p-1).
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------ f and g

class _CopyToTP(torch.autograd.Function):
    """f: identity forward, all-reduce gradient backward."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        # >>> SOLUTION: all_reduce the incoming gradient (sum) and return it
        g = g.clone()
        dist.all_reduce(g)
        return g
        # <<< SOLUTION


class _ReduceFromTP(torch.autograd.Function):
    """g: all-reduce forward, identity backward."""

    @staticmethod
    def forward(ctx, x):
        # >>> SOLUTION: all_reduce a copy of x (sum) and return it
        x = x.clone()
        dist.all_reduce(x)
        return x
        # <<< SOLUTION

    @staticmethod
    def backward(ctx, g):
        return g


copy_to_tp = _CopyToTP.apply
reduce_from_tp = _ReduceFromTP.apply


# ------------------------------------------------------------ TP layers

class ColumnParallelLinear(nn.Module):
    """Holds W[:, out_slice]. Output stays sharded along the feature dim."""

    def __init__(self, full: nn.Linear, rank: int, world: int):
        super().__init__()
        out = full.out_features // world
        self.weight = nn.Parameter(full.weight.detach()[rank * out : (rank + 1) * out].clone())
        self.bias = None if full.bias is None else nn.Parameter(full.bias.detach()[rank * out : (rank + 1) * out].clone())

    def forward(self, x):
        return F.linear(copy_to_tp(x), self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Holds W[in_slice, :]. Input arrives sharded; output is all-reduced."""

    def __init__(self, full: nn.Linear, rank: int, world: int):
        super().__init__()
        inp = full.in_features // world
        self.weight = nn.Parameter(full.weight.detach()[:, rank * inp : (rank + 1) * inp].clone())
        # Bias is added once, after the reduction, or it would be counted world times.
        self.bias = None if full.bias is None else nn.Parameter(full.bias.detach().clone())

    def forward(self, x_shard):
        # >>> SOLUTION: partial = x_shard @ W_shard^T; reduce_from_tp(partial); add bias after
        y = reduce_from_tp(F.linear(x_shard, self.weight))
        return y if self.bias is None else y + self.bias
        # <<< SOLUTION


class TPMLP(nn.Module):
    def __init__(self, up: nn.Linear, down: nn.Linear, rank: int, world: int):
        super().__init__()
        self.up = ColumnParallelLinear(up, rank, world)
        self.down = RowParallelLinear(down, rank, world)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class TPAttention(nn.Module):
    """Heads split across ranks. Built from full q, k, v, o projections."""

    def __init__(self, q: nn.Linear, k: nn.Linear, v: nn.Linear, o: nn.Linear, n_head: int, rank: int, world: int):
        super().__init__()
        assert n_head % world == 0
        self.local_heads = n_head // world
        self.head_dim = q.out_features // n_head
        self.q = ColumnParallelLinear(q, rank, world)
        self.k = ColumnParallelLinear(k, rank, world)
        self.v = ColumnParallelLinear(v, rank, world)
        self.o = RowParallelLinear(o, rank, world)

    def forward(self, x):
        B, T, _ = x.shape
        # >>> SOLUTION: project with column-parallel q/k/v -> (B, local_heads, T, hd); causal SDPA; merge heads; row-parallel o
        H, D = self.local_heads, self.head_dim
        q = self.q(x).view(B, T, H, D).transpose(1, 2)
        k = self.k(x).view(B, T, H, D).transpose(1, 2)
        v = self.v(x).view(B, T, H, D).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, H * D))
        # <<< SOLUTION


def full_attention(q, k, v, o, n_head, x):
    B, T, C = x.shape
    D = C // n_head
    sh = lambda t: t.view(B, T, n_head, D).transpose(1, 2)
    y = F.scaled_dot_product_attention(sh(q(x)), sh(k(x)), sh(v(x)), is_causal=True)
    return o(y.transpose(1, 2).reshape(B, T, C))


def tp_check(rank: int, world: int, d: int = 32, n_head: int = 4):
    """Return (max |out diff|, max |dx diff|) of TP vs single-device modules."""
    torch.manual_seed(0)
    up, down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
    q, k, v, o = (nn.Linear(d, d, bias=False) for _ in range(4))
    x = torch.randn(2, 6, d, requires_grad=True)
    ref = full_attention(q, k, v, o, n_head, x) + down(F.gelu(up(x)))
    (gx_ref,) = torch.autograd.grad(ref.pow(2).sum(), x)
    xt = x.detach().clone().requires_grad_(True)
    mlp = TPMLP(up, down, rank, world)
    attn = TPAttention(q, k, v, o, n_head, rank, world)
    out = attn(xt) + mlp(xt)
    (gx,) = torch.autograd.grad(out.pow(2).sum(), xt)
    return (out - ref).abs().max().item(), (gx - gx_ref).abs().max().item()


# ------------------------------------------------------------ pipeline

def bubble_fraction(stages: int, micro: int) -> float:
    # >>> SOLUTION: (p - 1) / (m + p - 1)
    return (stages - 1) / (micro + stages - 1)
    # <<< SOLUTION


def schedule_1f1b(stages: int, micro: int, stage: int) -> list[tuple[str, int]]:
    """Ordered ("F"|"B", microbatch) actions for one stage under 1F1B.

    warmup = min(stages - stage - 1, micro) forwards, then alternate one
    forward / one backward, then drain the remaining backwards."""
    # >>> SOLUTION: warmup forwards; steady (F, B) pairs; cooldown backwards
    warm = min(stages - stage - 1, micro)
    ops, f, b = [], 0, 0
    for _ in range(warm):
        ops.append(("F", f)); f += 1
    for _ in range(micro - warm):
        ops.append(("F", f)); f += 1
        ops.append(("B", b)); b += 1
    for _ in range(warm):
        ops.append(("B", b)); b += 1
    return ops
    # <<< SOLUTION


def make_stages(n_stages: int, d: int = 16, seed: int = 0) -> list[nn.Module]:
    torch.manual_seed(seed)
    return [nn.Sequential(nn.Linear(d, d), nn.Tanh()) for _ in range(n_stages)]


def pipeline_train_step(rank: int, world: int, micro: int = 4, d: int = 16):
    """One 1F1B step across `world` stages. Returns this stage's param grads.

    Sends are non-blocking (isend) so neighbouring stages never deadlock waiting
    for each other; receives are blocking because we need the data now."""
    stage = make_stages(world, d)[rank]
    g = torch.Generator().manual_seed(42)
    x_all = torch.randn(micro, 3, d, generator=g)
    y_all = torch.randn(micro, 3, d, generator=g)
    inputs, outputs, pending = {}, {}, []
    first, last = rank == 0, rank == world - 1
    for op, i in schedule_1f1b(world, micro, rank):
        if op == "F":
            # >>> SOLUTION: input = data (first) or recv from rank-1; forward; isend to rank+1 unless last
            if first:
                inp = x_all[i]
            else:
                inp = torch.empty(3, d)
                dist.recv(inp, rank - 1)
            inp.requires_grad_(not first)
            out = stage(inp)
            inputs[i], outputs[i] = inp, out
            if not last:
                pending.append(dist.isend(out.detach().contiguous(), rank + 1))
            # <<< SOLUTION
        else:
            # >>> SOLUTION: last: backward from loss/micro; else recv grad from rank+1 and backward; isend input grad to rank-1
            out = outputs.pop(i)
            if last:
                (F.mse_loss(out, y_all[i]) / micro).backward()
            else:
                gout = torch.empty(3, d)
                dist.recv(gout, rank + 1)
                out.backward(gout)
            inp = inputs.pop(i)
            if not first:
                pending.append(dist.isend(inp.grad.contiguous(), rank - 1))
            # <<< SOLUTION
    for req in pending:
        req.wait()
    return [p.grad.clone() for p in stage.parameters()]


def reference_grads(world: int, micro: int = 4, d: int = 16):
    stages = make_stages(world, d)
    model = nn.Sequential(*stages)
    g = torch.Generator().manual_seed(42)
    x_all = torch.randn(micro, 3, d, generator=g)
    y_all = torch.randn(micro, 3, d, generator=g)
    for i in range(micro):
        (F.mse_loss(model(x_all[i]), y_all[i]) / micro).backward()
    return [[p.grad.clone() for p in s.parameters()] for s in stages]


if __name__ == "__main__":
    from common import dist as ldist

    print("TP (out diff, dx diff):", ldist.run(tp_check, 2)[0])
    for p, m in [(4, 4), (4, 16), (8, 64)]:
        print(f"bubble p={p} m={m}: {bubble_fraction(p, m):.2%}")
    print("1F1B stage 0 of 4, m=6:", schedule_1f1b(4, 6, 0))
