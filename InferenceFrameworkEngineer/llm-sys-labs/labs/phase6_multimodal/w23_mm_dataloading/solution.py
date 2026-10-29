"""Week 23: data-reading acceleration for multimodal training.

Roadmap: "Sharded streaming datasets, image decode on CPU workers vs GPU
(nvJPEG/DALI), sequence packing of mixed image and text. Measure GPU idle time
from data stalls before and after."

Parts:
  1. Shards     - samples written to N shard files; images stored compressed
                  (zlib) so decoding costs real CPU time, like JPEG does
  2. Streaming  - an IterableDataset that splits shards across DDP ranks AND
                  DataLoader workers so every sample is read exactly once
  3. Stall time - time the training loop spends waiting on next(loader),
                  with 0 workers vs several
  4. Packing    - first-fit-decreasing packing of variable-length samples
                  into fixed rows, with a block-diagonal causal mask and
                  per-sample position ids; packed logits must equal unpacked

GPU note: on a real cluster move decode to the GPU (nvJPEG via
torchvision.io.decode_jpeg(device="cuda") or NVIDIA DALI) and compare stall
time again; the measurement code here is unchanged.
"""
from __future__ import annotations

import math
import os
import pathlib
import pickle
import sys
import tempfile
import time
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


# ------------------------------------------------------------------ 1. shards


def write_shards(n_samples: int, n_shards: int, out_dir: str, img: int = 64, seed: int = 0) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    paths = []
    per = math.ceil(n_samples / n_shards)
    for s in range(n_shards):
        rows = []
        for i in range(s * per, min(n_samples, (s + 1) * per)):
            pix = (torch.rand(3, img, img, generator=g) * 255).to(torch.uint8).numpy().tobytes()
            text_len = int(torch.randint(8, 64, (1,), generator=g))
            rows.append({"id": i, "img": zlib.compress(pix, 6), "shape": (3, img, img),
                         "text": torch.randint(0, 256, (text_len,), generator=g).tolist()})
        p = os.path.join(out_dir, f"shard-{s:05d}.pkl")
        with open(p, "wb") as f:
            pickle.dump(rows, f)
        paths.append(p)
    return paths


def decode(row, out_size: int = 224) -> torch.Tensor:
    """The CPU-heavy step: decompress (JPEG decode in real life) + antialiased
    resize to the vision tower's input size, as an image pipeline does."""
    raw = zlib.decompress(row["img"])
    img = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(*row["shape"]).float().div_(255)
    out = torch.nn.functional.interpolate(img[None], size=(out_size, out_size), mode="bilinear",
                                          antialias=True, align_corners=False)[0]
    # Return uint8: a float32 224x224 batch of 32 is ~19 MB to ship from a worker
    # process to the trainer; uint8 is 4x smaller. Convert to float on the device.
    return out.mul_(255).round_().to(torch.uint8)


# ------------------------------------------------------------------ 2. streaming


class ShardStream(IterableDataset):
    def __init__(self, shards: list[str], rank: int = 0, world: int = 1):
        self.shards, self.rank, self.world = shards, rank, world

    def my_shards(self) -> list[str]:
        """Shards this (rank, worker) pair owns: disjoint across all pairs."""
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        # >>> SOLUTION: global reader index = rank * nw + wid out of world * nw readers; take every (world*nw)-th shard starting there
        reader, readers = self.rank * nw + wid, self.world * nw
        return self.shards[reader::readers]
        # <<< SOLUTION

    def __iter__(self):
        for p in self.my_shards():
            with open(p, "rb") as f:
                for row in pickle.load(f):
                    yield {"id": row["id"], "img": decode(row), "text": torch.tensor(row["text"])}


def worker_init(_):
    """One intra-op thread per loader worker. Without this every worker spins up
    PyTorch's full thread pool and they fight the trainer for the same cores."""
    torch.set_num_threads(1)


def collate(rows):
    """Return a few large tensors, not lists of small ones: every tensor in a
    batch crosses the worker -> trainer boundary as its own shared-memory
    segment, so 32 caption tensors cost 32 segment hand-offs per step."""
    lens = torch.tensor([len(r["text"]) for r in rows])
    text = torch.zeros(len(rows), int(lens.max()), dtype=torch.long)
    for i, r in enumerate(rows):
        text[i, : len(r["text"])] = r["text"]
    return {"id": torch.tensor([r["id"] for r in rows]), "img": torch.stack([r["img"] for r in rows]),
            "text": text, "text_len": lens}


# ------------------------------------------------------------------ 3. stall time


def measure_stall(shards, workers: int, batch: int = 32, step_compute_s: float = 0.02) -> dict:
    """Simulated train step of fixed compute; report time blocked on data.

    The first batch is reported separately as start-up (worker processes
    spawning and filling their prefetch queues); "data wait s" is the
    steady-state stall that matters over a long run."""
    loader = DataLoader(ShardStream(shards), batch_size=batch, num_workers=workers, collate_fn=collate,
                        prefetch_factor=4 if workers else None, persistent_workers=False,
                        worker_init_fn=worker_init if workers else None)
    wait = startup = 0.0
    n = 0
    t_start = time.perf_counter()
    it = iter(loader)
    while True:
        t0 = time.perf_counter()
        # >>> SOLUTION: time next(it) as data wait (stop on StopIteration); then "compute" for step_compute_s (busy loop or sleep)
        try:
            next(it)
        except StopIteration:
            break
        if n == 0:
            startup = time.perf_counter() - t0
        else:
            wait += time.perf_counter() - t0
        time.sleep(step_compute_s)  # stands in for the GPU step; sleeping frees the CPU for workers
        n += 1
        # <<< SOLUTION
    total = time.perf_counter() - t_start
    steady = total - startup
    return {"workers": workers, "steps": n, "startup s": startup, "data wait s": wait, "total s": total,
            "stall %": 100 * wait / steady}


def decode_cost_per_batch(shards, batch: int = 32) -> float:
    """Single-threaded seconds to decode one batch (what one loader worker pays)."""
    with open(shards[0], "rb") as f:
        rows = pickle.load(f)[:batch]
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    for r in rows:
        decode(r)
    torch.set_num_threads(prev)
    return (time.perf_counter() - t0) * batch / len(rows)


def workers_needed(decode_s_per_batch: float, step_s: float, cores: int) -> tuple[int, bool]:
    """Workers to hide decode behind compute, and whether this machine has the cores.

    Each worker supplies one batch per decode_s; the trainer consumes one per
    step_s. Workers also need cores the trainer is not using."""
    # >>> SOLUTION: need = ceil(decode / step); feasible if need <= cores - 1 (one core for the trainer's host work)
    need = math.ceil(decode_s_per_batch / step_s)
    return need, need <= max(cores - 1, 0)
    # <<< SOLUTION


# ------------------------------------------------------------------ 4. packing


def pack_ffd(lengths: list[int], capacity: int) -> list[list[int]]:
    """First-fit-decreasing: indices of samples per row; sum of lengths <= capacity."""
    # >>> SOLUTION: sort indices by length descending; place each into the first row with room, else open a new row
    rows: list[list[int]] = []
    room: list[int] = []
    for i in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        for r in range(len(rows)):
            if room[r] >= lengths[i]:
                rows[r].append(i)
                room[r] -= lengths[i]
                break
        else:
            rows.append([i])
            room.append(capacity - lengths[i])
    return rows
    # <<< SOLUTION


def build_packed(seqs: list[torch.Tensor], groups: list[list[int]], capacity: int, pad: int = 0):
    """Return (tokens (R, C), pos (R, C), mask (R, C, C), seg (R, C)) for packed rows.

    mask[r, i, j] is True when token j may be attended by token i: same sample
    and j <= i. Positions restart at 0 for every sample, so RoPE sees each
    sample exactly as it would alone."""
    R = len(groups)
    tok = torch.full((R, capacity), pad, dtype=torch.long)
    pos = torch.zeros(R, capacity, dtype=torch.long)
    seg = torch.full((R, capacity), -1, dtype=torch.long)
    for r, g in enumerate(groups):
        off = 0
        for i in g:
            n = len(seqs[i])
            # >>> SOLUTION: write tokens at [off, off+n), positions 0..n-1, segment id i; advance off
            tok[r, off : off + n] = seqs[i]
            pos[r, off : off + n] = torch.arange(n)
            seg[r, off : off + n] = i
            off += n
            # <<< SOLUTION
    same = (seg[:, :, None] == seg[:, None, :]) & (seg[:, :, None] >= 0)
    causal = torch.ones(capacity, capacity, dtype=torch.bool).tril()
    mask = same & causal
    mask |= torch.eye(capacity, dtype=torch.bool)  # padding attends to itself only (avoids all-masked rows)
    return tok, pos, mask, seg


def padding_efficiency(lengths: list[int], batch: int) -> float:
    """Useful tokens / total when batching in arrival order, padding to the batch max."""
    useful = total = 0
    for i in range(0, len(lengths), batch):
        b = lengths[i : i + batch]
        useful += sum(b)
        total += max(b) * len(b)
    return useful / total


def packing_efficiency(lengths: list[int], capacity: int) -> float:
    rows = pack_ffd(lengths, capacity)
    return sum(lengths) / (len(rows) * capacity)


if __name__ == "__main__":
    from common.bench import table

    d = tempfile.mkdtemp()
    shards = write_shards(2048, 8, d)
    dec_s = decode_cost_per_batch(shards)
    for step in (0.02, 0.08):
        need, ok = workers_needed(dec_s, step, os.cpu_count() or 1)
        print(f"step {step*1e3:.0f} ms, decode {dec_s*1e3:.0f} ms/batch/worker -> need {need} workers; "
              f"{'feasible' if ok else 'NOT feasible'} on {os.cpu_count()} cores")
        print(table([measure_stall(shards, w, step_compute_s=step) for w in (0, 2)],
                    ["workers", "steps", "startup s", "data wait s", "total s", "stall %"]))
    g = torch.Generator().manual_seed(0)
    lengths = [int(torch.randint(16 + 8, 16 + 64, (1,), generator=g)) for _ in range(2000)]  # 16 image tokens + text
    print(f"token efficiency: padded batches {padding_efficiency(lengths, 32):.1%} vs FFD packing {packing_efficiency(lengths, 256):.1%}")
