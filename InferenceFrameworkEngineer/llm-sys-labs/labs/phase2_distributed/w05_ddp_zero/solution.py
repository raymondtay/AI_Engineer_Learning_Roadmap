"""Week 5: collectives, data parallel by hand, ZeRO.

Roadmap: "Run nccl-tests; learn all-reduce, all-gather, reduce-scatter,
all-to-all costs. Write DDP by hand, then use FSDP2 and ZeRO stages 1-3."

CPU: N processes with gloo. GPU: one process per GPU with NCCL. Same code.

Parts:
  1. bus_bandwidth   - the nccl-tests formula for all-reduce
  2. ddp_step        - bucketed gradient all-reduce, by hand
  3. ZeroAdam        - ZeRO-2: reduce-scatter grads, Adam on my shard, all-gather params
  4. fsdp2 (GPU lab) - torch.distributed.fsdp.fully_shard on the same model

Run:  python solution.py            # 2-process CPU demo + collective bandwidth
Real cluster:  torchrun --nproc-per-node 8 solution.py --torchrun
"""
from __future__ import annotations

import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist
import torch.nn as nn

from common import dist as ldist


# ------------------------------------------------------------------ part 1

def bus_bandwidth(nbytes: int, seconds: float, world: int) -> float:
    """All-reduce bus bandwidth (bytes/s) as nccl-tests reports it.

    A ring all-reduce moves 2*(n-1)/n of the buffer per rank, so
    busbw = algbw * 2(n-1)/n, with algbw = bytes / time."""
    # >>> SOLUTION: algbw = nbytes/seconds; multiply by 2*(world-1)/world
    return nbytes / seconds * 2 * (world - 1) / world
    # <<< SOLUTION


def allreduce_bench(rank: int, world: int, sizes=(2**16, 2**20, 2**22)) -> list[dict]:
    dev = ldist.device_for_rank(rank)
    rows = []
    for n in sizes:
        x = torch.ones(n // 4, device=dev)
        for _ in range(2):
            dist.all_reduce(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            dist.all_reduce(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / iters
        rows.append({"bytes": n, "us": t * 1e6, "busbw GB/s": bus_bandwidth(n, t, world) / 1e9})
    return rows


# ------------------------------------------------------------------ part 2

def make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 8))


def make_data(seed: int = 1, n: int = 64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 32, generator=g), torch.randn(n, 8, generator=g)


def ddp_allreduce_grads(model: nn.Module, world: int, bucket_bytes: int = 4096) -> int:
    """Average gradients across ranks with bucketed all-reduce.

    Buckets fuse many small tensors into one collective (fewer launches, better
    bandwidth); real DDP also overlaps each bucket with the rest of backward.
    Returns the number of all-reduce calls issued."""
    calls = 0
    # >>> SOLUTION: group grads into buckets up to bucket_bytes, flatten, all_reduce, divide by world, copy back
    bucket: list[torch.Tensor] = []
    size = 0

    def flush():
        nonlocal bucket, size, calls
        if not bucket:
            return
        flat = torch.cat([g.flatten() for g in bucket])
        dist.all_reduce(flat)
        flat /= world
        off = 0
        for g in bucket:
            g.copy_(flat[off : off + g.numel()].view_as(g))
            off += g.numel()
        calls += 1
        bucket, size = [], 0

    for p in reversed(list(model.parameters())):  # backward produces grads last layer first
        g = p.grad
        bucket.append(g)
        size += g.numel() * g.element_size()
        if size >= bucket_bytes:
            flush()
    flush()
    # <<< SOLUTION
    return calls


def bucket_call_counts(rank: int, world: int):
    """(calls with tiny buckets, calls with 1 MB buckets) for the same backward."""
    m = make_model()
    x, y = make_data()
    nn.functional.mse_loss(m(x), y).backward()
    return ddp_allreduce_grads(m, world, bucket_bytes=1), ddp_allreduce_grads(m, world, bucket_bytes=1 << 20)


def ddp_train(rank: int, world: int, steps: int = 5, lr: float = 1e-2):
    model = make_model()
    x, y = make_data()
    shard = x.shape[0] // world
    xs, ys = x[rank * shard : (rank + 1) * shard], y[rank * shard : (rank + 1) * shard]
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(xs), ys)
        loss.backward()
        ddp_allreduce_grads(model, world)
        opt.step()
        opt.zero_grad()
    return [p.detach().clone() for p in model.parameters()]


# ------------------------------------------------------------------ part 3

class ZeroAdam:
    """ZeRO stage 2 over a flat parameter buffer.

    Every rank keeps full params (for forward/backward) but only 1/world of the
    Adam state (m, v) and only reduces 1/world of the gradient:
      1. flatten grads, reduce_scatter (sum) -> my shard of the summed grad
      2. Adam update on my shard of the params
      3. all_gather the updated shards -> full params on every rank
    Optimizer memory per rank drops by `world`; communication equals one
    all-reduce (reduce-scatter + all-gather is exactly how ring all-reduce works)."""

    def __init__(self, params, world: int, rank: int, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params]
        self.world, self.rank = world, rank
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.numel = sum(p.numel() for p in self.params)
        self.padded = math.ceil(self.numel / world) * world
        self.shard = self.padded // world
        dev = self.params[0].device
        self.m = torch.zeros(self.shard, device=dev)
        self.v = torch.zeros(self.shard, device=dev)
        self.t = 0

    def _flat(self, tensors) -> torch.Tensor:
        flat = torch.cat([t.flatten() for t in tensors])
        return torch.nn.functional.pad(flat, (0, self.padded - self.numel))

    @torch.no_grad()
    def step(self):
        self.t += 1
        lo = self.rank * self.shard
        # >>> SOLUTION: reduce_scatter_tensor(sum) the flat grads, average, Adam on my param slice, all_gather_into_tensor, unflatten
        g_full = self._flat([p.grad for p in self.params])
        g = torch.empty(self.shard, device=g_full.device)
        dist.reduce_scatter_tensor(g, g_full, op=dist.ReduceOp.SUM)
        g /= self.world
        p_full = self._flat([p.detach() for p in self.params])
        p = p_full[lo : lo + self.shard].clone()
        self.m.mul_(self.b1).add_(g, alpha=1 - self.b1)
        self.v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        p -= self.lr * mhat / (vhat.sqrt() + self.eps)
        dist.all_gather_into_tensor(p_full, p)
        off = 0
        for prm in self.params:
            prm.copy_(p_full[off : off + prm.numel()].view_as(prm))
            off += prm.numel()
        # <<< SOLUTION

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_bytes(self) -> int:
        return (self.m.numel() + self.v.numel()) * 4


def zero_train(rank: int, world: int, steps: int = 5, lr: float = 1e-2):
    model = make_model()
    x, y = make_data()
    shard = x.shape[0] // world
    xs, ys = x[rank * shard : (rank + 1) * shard], y[rank * shard : (rank + 1) * shard]
    opt = ZeroAdam(model.parameters(), world, rank, lr=lr)
    for _ in range(steps):
        nn.functional.mse_loss(model(xs), ys).backward()
        opt.step()
        opt.zero_grad()
    return [p.detach().clone() for p in model.parameters()], opt.state_bytes()


def reference_train(optim: str, steps: int = 5, lr: float = 1e-2):
    """Single-process, full batch. DDP and ZeRO must match this."""
    model = make_model()
    x, y = make_data()
    opt = torch.optim.SGD(model.parameters(), lr=lr) if optim == "sgd" else torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()
        opt.zero_grad()
    return [p.detach().clone() for p in model.parameters()]


# ------------------------------------------------------------------ part 4

def fsdp2_train(rank: int, world: int, steps: int = 5, lr: float = 1e-2):
    """FSDP2: the production version of ZeRO-3 (params sharded too)."""
    from torch.distributed.fsdp import fully_shard

    model = make_model().to(ldist.device_for_rank(rank))
    for layer in model:
        if isinstance(layer, nn.Linear):
            fully_shard(layer)
    fully_shard(model)
    x, y = (t.to(ldist.device_for_rank(rank)) for t in make_data())
    shard = x.shape[0] // world
    xs, ys = x[rank * shard : (rank + 1) * shard], y[rank * shard : (rank + 1) * shard]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        nn.functional.mse_loss(model(xs), ys).backward()
        opt.step()
        opt.zero_grad()
    return [p.full_tensor().detach().cpu() for p in model.parameters()]


if __name__ == "__main__":
    from common.bench import table

    rows = ldist.run(allreduce_bench, 2)[0]
    print("all-reduce, 2 ranks:\n" + table(rows, ["bytes", "us", "busbw GB/s"]))
    ref = reference_train("adam")
    ours, state = ldist.run(zero_train, 2)[0]
    err = max((a - b).abs().max().item() for a, b in zip(ref, ours))
    print(f"ZeRO-2 vs single-process Adam: max |diff| = {err:.2e}; Adam state per rank = {state} bytes")
