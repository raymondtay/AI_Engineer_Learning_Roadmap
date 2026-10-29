# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 16: KV cache to CPU and disk - a tiered store with LRU and async prefetch.

Roadmap: "Run LMCache with vLLM for CPU then local-SSD offload. Build your own
tiered store (GPU, DRAM, NVMe) with LRU and async prefetch."
Gate: measured TTFT reduction on a multi-turn trace AND the overhead on a
no-reuse trace, both reported.

  tier 0  "device"  - GPU memory on CUDA; a bounded CPU dict on CPU-only machines
  tier 1  "host"    - CPU DRAM (pinned on CUDA so copies can be asynchronous)
  tier 2  "disk"    - one file per block under a directory (NVMe in production)

Blocks enter at tier 0. When a tier is full its least-recently-used block is
demoted one tier down; a hit in a lower tier promotes the block back to tier 0.
`prefetch` starts disk -> host loads on a thread pool so the next request's
blocks are warm before the scheduler needs them (what Mooncake and LMCache do
over RDMA/NVMe in production).

The KVService at the bottom puts this behind the block-hash lookup from Week 15
and runs real prefills on the shared tiny GPT.
"""
from __future__ import annotations

import pathlib
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from common.device import device
from common.tinygpt import GPT, GPTConfig

TIERS = ("device", "host", "disk")


class TieredKVStore:
    def __init__(self, cap_device: int, cap_host: int, cap_disk: int, disk_dir: str | None = None):
        self.caps = {"device": cap_device, "host": cap_host, "disk": cap_disk}
        self.tiers: dict[str, OrderedDict] = {t: OrderedDict() for t in TIERS}
        self.dir = pathlib.Path(disk_dir or tempfile.mkdtemp(prefix="kvstore_"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dev = device()
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.pending: dict[int, Future] = {}
        self.stats = {f"hit_{t}": 0 for t in TIERS} | {"miss": 0, "demotions": 0, "promotions": 0}

    # ---------------- representation per tier
    def _to_tier(self, key: int, value, tier: str):
        """Convert a block's value to the representation used by `tier`."""
        # TODO: device -> tensor on self.dev; host -> CPU tensor (pin_memory on CUDA); disk -> torch.save to dir/key.pt and store the path
        raise NotImplementedError("TODO(w16_tiered_kv): device -> tensor on self.dev; host -> CPU tensor (pin_memory on CUDA); disk -> torch.save to dir/key.pt and store the path")

    def _insert(self, key: int, value, tier_idx: int):
        """Place at tier_idx (most recent), demoting LRU blocks down as needed."""
        tier = TIERS[tier_idx]
        # TODO: store converted value as MRU; while over capacity pop LRU and recurse into tier_idx+1 (drop it past the last tier, deleting any disk file)
        raise NotImplementedError("TODO(w16_tiered_kv): store converted value as MRU; while over capacity pop LRU and recurse into tier_idx+1 (drop it past the last tier, deleting any disk file)")

    # ---------------- public API
    def put(self, key: int, block: torch.Tensor):
        with self.lock:
            self._remove(key)
            self._insert(key, block, 0)

    def get(self, key: int) -> torch.Tensor | None:
        fut = self.pending.pop(key, None)
        if fut is not None:
            fut.result()  # a prefetch for this key is in flight; wait for it
        with self.lock:
            for t in TIERS:
                if key in self.tiers[t]:
                    self.stats[f"hit_{t}"] += 1
                    val = self.tiers[t].pop(key)
                    if t == "device":
                        self.tiers[t][key] = val
                        return val
                    # TODO: promote: convert to device form, insert at tier 0, delete the disk file if it came from disk; return the tensor
                    raise NotImplementedError("TODO(w16_tiered_kv): promote: convert to device form, insert at tier 0, delete the disk file if it came from disk; return the tensor")
            self.stats["miss"] += 1
            return None

    def contains(self, key: int) -> bool:
        with self.lock:
            return any(key in self.tiers[t] for t in TIERS)

    def prefetch(self, keys: list[int]):
        """Asynchronously lift disk-resident blocks into host memory."""
        for k in keys:
            with self.lock:
                on_disk = k in self.tiers["disk"] and k not in self.pending
            if on_disk:
                self.pending[k] = self.pool.submit(self._lift, k)

    def _lift(self, key: int):
        with self.lock:
            path = self.tiers["disk"].pop(key, None)
        if path is None:
            return
        t = torch.load(path)  # the slow part runs outside the lock
        with self.lock:
            path.unlink(missing_ok=True)
            self._insert(key, t, 1)

    def where(self, key: int) -> str | None:
        with self.lock:
            return next((t for t in TIERS if key in self.tiers[t]), None)

    def _remove(self, key: int):
        for t in TIERS:
            v = self.tiers[t].pop(key, None)
            if isinstance(v, pathlib.Path):
                v.unlink(missing_ok=True)

    def close(self):
        self.pool.shutdown(wait=True)
        shutil.rmtree(self.dir, ignore_errors=True)


# ------------------------------------------------------------------ KV service


def block_keys(tokens: list[int], B: int) -> list[int]:
    out, h = [], None
    for i in range(len(tokens) // B):
        h = hash((h, tuple(tokens[i * B : (i + 1) * B])))
        out.append(h)
    return out


class KVService:
    """Prefill with reuse: find the longest stored block prefix, load it from
    whichever tier holds it, compute only the rest, store the new full blocks."""

    def __init__(self, model: GPT, store: TieredKVStore | None, block_size: int = 16):
        self.model, self.store, self.B = model.eval(), store, block_size

    def prefill(self, prompt: list[int]):
        dev = next(self.model.parameters()).device
        keys = block_keys(prompt, self.B)
        hit_blocks = []
        if self.store is not None:
            for k in keys:
                if (len(hit_blocks) + 1) * self.B >= len(prompt):
                    break  # leave at least one token to compute
                blk = self.store.get(k)
                if blk is None:
                    break
                hit_blocks.append(blk)
        hit = len(hit_blocks) * self.B
        past = None
        if hit_blocks:
            stacked = torch.cat([b.to(dev) for b in hit_blocks], dim=-2)  # (L, 2, H, hit, D)
            past = [(stacked[l, 0][None], stacked[l, 1][None]) for l in range(stacked.shape[0])]
        logits, new_kv = self.model.forward_with_past(torch.tensor(prompt[hit:], device=dev), hit, past)
        if self.store is not None:
            kv = torch.stack([torch.stack([k[0], v[0]]) for k, v in new_kv])  # (L, 2, H, T_new, D)
            for bi in range(hit // self.B, len(prompt) // self.B):
                s = bi * self.B - hit
                self.store.put(keys[bi], kv[..., s : s + self.B, :].clone())
        return logits, hit


def ttft_report(trace, caps=(64, 256, 4096), block_size: int = 16, d_model: int = 384) -> dict:
    torch.manual_seed(0)
    model = GPT(GPTConfig(seq_len=2048, n_layer=4, n_head=6, d_model=d_model)).to(device()).eval()
    model.forward_with_past(torch.tensor(trace[0][:32], device=device()), 0)  # warm-up
    out = {}
    for name, store in [("no cache", None), ("tiered", TieredKVStore(*caps))]:
        svc = KVService(model, store, block_size)
        times, hits = [], 0
        for i, p in enumerate(trace):
            if store is not None and i + 1 < len(trace):
                store.prefetch(block_keys(trace[i + 1], block_size))
            t0 = time.perf_counter()
            _, h = svc.prefill(p)
            times.append(time.perf_counter() - t0)
            hits += h
        out[name] = {"median TTFT ms": 1e3 * statistics.median(times), "tokens reused": hits}
        if store is not None:
            out[name] |= store.stats
            store.close()
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "w15_prefix_cache"))
    from solution import multi_turn_trace, no_reuse_trace  # noqa: E402  (Week 15 traces)

    mt = multi_turn_trace(n_sessions=6, turns=4)
    print("multi-turn:", ttft_report(mt, caps=(8, 32, 4096)))
    print("no-reuse:  ", ttft_report(no_reuse_trace(n=16)))
