# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 17: prefill/decode disaggregation and cache-aware routing.

Roadmap: "Split prefill and decode across two instances, transfer KV via
Mooncake Transfer Engine or NIXL. Write a router that scores instances on
cached-prefix length and queue depth."
Gate: cache-aware router vs round-robin on 2+ instances - P50/P99 TTFT and throughput.

Part A - PD disaggregation (real processes). Rank 0 is a prefill worker, rank 1
a decode worker. The prefill worker computes the prompt's KV and first token and
ships the KV with point-to-point sends (gloo on CPU, NCCL on GPU; Mooncake and
NIXL do the same over RDMA). The decode worker continues generation. Output
must equal a single-process run, and we account for the bytes moved.

Part B - routing (discrete-event simulation, service times calibrated on the
real tiny GPT). Each instance keeps a prefix cache. Policies:
  round_robin     - ignores everything
  least_loaded    - shortest queue (in seconds of queued work)
  cache_aware     - minimise estimated TTFT = queue wait + prefill(uncached tokens on that instance)
Why simulate: the effect depends on arrival rate and cache hit structure; a
simulator lets you sweep both in seconds. On GPU, repeat with real servers.
"""
from __future__ import annotations

import heapq
import pathlib
import random
import statistics
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist

from common import dist as ldist
from common.tinygpt import GPT, GPTConfig


# ------------------------------------------------------------------ A. PD disaggregation


def make_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(seq_len=512, n_layer=2, n_head=4, d_model=64)).eval()


def requests(n=4, seed=0):
    rnd = random.Random(seed)
    return [[rnd.randrange(256) for _ in range(rnd.randrange(10, 60))] for _ in range(n)]


def greedy_monolithic(model, prompt, n):
    logits, kv = model.forward_with_past(torch.tensor(prompt), 0)
    out = [int(logits.argmax())]
    past = kv
    for i in range(n - 1):
        logits, new = model.forward_with_past(torch.tensor([out[-1]]), len(prompt) + i, past)
        past = [(torch.cat([pk, nk], 2), torch.cat([pv, nv], 2)) for (pk, pv), (nk, nv) in zip(past, new)]
        out.append(int(logits.argmax()))
    return out


def pd_worker(rank: int, world: int, n_new: int = 8):
    """rank 0 = prefill, rank 1 = decode. Returns decode outputs (rank 1) or bytes sent (rank 0)."""
    model = make_model()
    reqs = requests()
    if rank == 0:
        sent = 0
        for prompt in reqs:
            # TODO: prefill; stack KV to (L, 2, H, T, D); send header [T, first_token], then the KV tensor; count bytes
            raise NotImplementedError("TODO(w17_pd_router): prefill; stack KV to (L, 2, H, T, D); send header [T, first_token], then the KV tensor; count bytes")
        return sent
    cfg = model.cfg
    outs = []
    for _ in reqs:
        # TODO: recv header, allocate (L, 2, H, T, D) and recv KV; rebuild past; decode n_new-1 more tokens greedily
        raise NotImplementedError("TODO(w17_pd_router): recv header, allocate (L, 2, H, T, D) and recv KV; rebuild past; decode n_new-1 more tokens greedily")
        outs.append(out)
    return outs


def kv_bytes_per_token(n_layer: int, n_kv_heads: int, head_dim: int, bytes_per_el: int = 2) -> int:
    """KV bytes the prefill worker must ship per prompt token."""
    # TODO: 2 (K and V) * layers * kv_heads * head_dim * bytes
    raise NotImplementedError("TODO(w17_pd_router): 2 (K and V) * layers * kv_heads * head_dim * bytes")


# ------------------------------------------------------------------ B. routing


@dataclass
class Req:
    rid: int
    arrival: float
    tokens: list
    gen: int
    session: int


def session_trace(n_sessions=40, turns=5, rate=20.0, sys_len=128, msg=32, reply=48, gen=48, seed=0):
    """Poisson arrivals of multi-turn chat requests; turns of a session arrive in order."""
    rnd = random.Random(seed)
    system = [rnd.randrange(256) for _ in range(sys_len)]
    hist = {s: list(system) for s in range(n_sessions)}
    order = [s for _ in range(turns) for s in rnd.sample(range(n_sessions), n_sessions)]
    t, out = 0.0, []
    for i, s in enumerate(order):
        t += rnd.expovariate(rate)
        hist[s] += [rnd.randrange(256) for _ in range(msg)]
        out.append(Req(i, t, list(hist[s]), gen, s))
        hist[s] += [rnd.randrange(256) for _ in range(reply)]
    return out


class PrefixLRU:
    """Per-instance prefix cache at block granularity (enough for routing decisions)."""

    def __init__(self, cap_blocks: int, B: int = 16):
        self.B, self.cap, self.d = B, cap_blocks, OrderedDict()

    def _keys(self, toks):
        h, out = None, []
        for i in range(len(toks) // self.B):
            h = hash((h, tuple(toks[i * self.B : (i + 1) * self.B])))
            out.append(h)
        return out

    def peek(self, toks) -> int:
        n = 0
        for k in self._keys(toks):
            if k not in self.d:
                break
            n += self.B
        return n

    def admit(self, toks):
        for k in self._keys(toks):
            self.d[k] = True
            self.d.move_to_end(k)
        while len(self.d) > self.cap:
            self.d.popitem(last=False)


@dataclass
class Costs:
    prefill_per_token: float = 2.0e-4  # seconds; calibrate() measures these on the real model
    decode_per_token: float = 1.0e-3
    overhead: float = 1.0e-3


def calibrate(d_model: int = 256) -> Costs:
    """Measure prefill cost per token and decode cost per token on the tiny GPT."""
    torch.manual_seed(0)
    m = GPT(GPTConfig(seq_len=2048, n_layer=4, n_head=4, d_model=d_model)).eval()
    toks = torch.randint(0, 256, (512,))
    m.forward_with_past(toks[:32], 0)
    t0 = time.perf_counter()
    _, kv = m.forward_with_past(toks, 0)
    pre = (time.perf_counter() - t0) / 512
    t0 = time.perf_counter()
    for i in range(16):  # one decode step at a fixed 512-token context
        m.forward_with_past(toks[i : i + 1], 512, kv)
    dec = (time.perf_counter() - t0) / 16
    return Costs(pre, dec, dec)


def route(policy: str, req: Req, instances: list[PrefixLRU], free_at: list[float], now: float,
          costs: Costs, rr: list[int]) -> int:
    # TODO: round_robin cycles rr[0]; least_loaded = argmin max(free_at - now, 0); cache_aware = argmin wait + prefill_per_token * (len - cached)
    raise NotImplementedError("TODO(w17_pd_router): round_robin cycles rr[0]; least_loaded = argmin max(free_at - now, 0); cache_aware = argmin wait + prefill_per_token * (len - cached)")


def simulate(trace: list[Req], n_inst: int, policy: str, costs: Costs, cache_blocks: int = 2000) -> dict:
    """Each instance serves one request at a time (FIFO): TTFT = wait + prefill of uncached tokens."""
    instances = [PrefixLRU(cache_blocks) for _ in range(n_inst)]
    free_at = [0.0] * n_inst
    rr = [-1]
    ttft, reused, total = [], 0, 0
    end = 0.0
    for req in trace:
        i = route(policy, req, instances, free_at, req.arrival, costs, rr)
        cached = instances[i].peek(req.tokens)
        start = max(req.arrival, free_at[i])
        first = start + costs.overhead + costs.prefill_per_token * (len(req.tokens) - cached)
        free_at[i] = first + costs.decode_per_token * (req.gen - 1)
        instances[i].admit(req.tokens)
        ttft.append(first - req.arrival)
        reused += cached
        total += len(req.tokens)
        end = max(end, free_at[i])
    q = statistics.quantiles(ttft, n=100)
    return {"policy": policy, "P50 TTFT ms": 1e3 * statistics.median(ttft), "P99 TTFT ms": 1e3 * q[98],
            "req/s": len(trace) / end, "prefix reuse": reused / total}


def compare(n_inst: int = 4, costs: Costs | None = None, **trace_kw) -> list[dict]:
    costs = costs or Costs()
    tr = session_trace(**trace_kw)
    return [simulate(tr, n_inst, p, costs) for p in ("round_robin", "least_loaded", "cache_aware")]


if __name__ == "__main__":
    from common.bench import table

    out = ldist.run(pd_worker, 2)
    model = make_model()
    ok = all(o == greedy_monolithic(model, p, 8) for o, p in zip(out[1], requests()))
    print(f"PD disaggregation matches monolithic: {ok}; KV bytes shipped: {out[0]}")
    print(f"KV per token, Llama-3-8B shape (32 layers, 8 KV heads, d=128, bf16): {kv_bytes_per_token(32, 8, 128)} B")
    c = calibrate()
    print(f"calibrated on this machine: prefill {c.prefill_per_token*1e6:.0f} us/token, decode {c.decode_per_token*1e3:.2f} ms/token")
    # Offered load ~60% of capacity: a request costs roughly ~300 prompt tokens of prefill + 48 decode steps.
    service = c.overhead + 300 * c.prefill_per_token + 47 * c.decode_per_token
    print(table(compare(costs=c, rate=0.6 * 4 / service),
                ["policy", "P50 TTFT ms", "P99 TTFT ms", "req/s", "prefix reuse"]))
