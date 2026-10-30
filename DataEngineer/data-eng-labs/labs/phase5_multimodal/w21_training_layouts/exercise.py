# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 21: storage layouts for training, resumable shuffles, and mixtures.

Roadmap: "Loose files vs tar shards (WebDataset) vs Parquet; deterministic,
resumable shuffling; mixture sampling with per-source weights and epoch caps."

1. Layout. Millions of small files make object stores and file systems pay a
   per-file cost (open, metadata, request). WebDataset packs samples into POSIX
   tar shards read sequentially; files sharing a basename form one sample
   (000123.txt + 000123.json). Parquet gives columnar projection and random
   access by row group. You measure all three.
2. Resumable shuffle. Training restarts after node failures; the data stream must
   continue exactly where it stopped or the model sees some samples twice and
   others never. Shard order is permuted per epoch, samples pass through a
   shuffle buffer, and the whole state (epoch, shard cursor, buffer, RNG) is
   checkpointable.
3. Mixtures. Pre-training draws from sources by weight (web, code, books...).
   Small sources run out and get repeated. Muennighoff et al. (NeurIPS 2023,
   "Scaling Data-Constrained Language Models"): "training with up to 4 epochs of
   repeated data yields negligible changes to loss compared to having unique
   data. However, with more repetition, the value of adding compute eventually
   decays to zero." The sampler reports
   epochs per source so you can enforce a cap.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import random
import sys
import tarfile
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table


def samples(n: int, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    return [{"key": f"{i:07d}", "text": " ".join(rnd.choice("abcdefgh") * rnd.randint(1, 8) for _ in range(rnd.randint(50, 300))),
             "meta": {"src": rnd.choice(["web", "code", "books"]), "i": i}} for i in range(n)]


# ------------------------------------------------------------------ 1. layouts

def write_files(ss, d):
    os.makedirs(d, exist_ok=True)
    for s in ss:
        pathlib.Path(d, s["key"] + ".txt").write_text(s["text"])
        pathlib.Path(d, s["key"] + ".json").write_text(json.dumps(s["meta"]))


def read_files(d):
    out = []
    for p in sorted(pathlib.Path(d).glob("*.txt")):
        out.append({"key": p.stem, "text": p.read_text(), "meta": json.loads(p.with_suffix(".json").read_text())})
    return out


def write_tar(ss, d, per_shard: int = 1000) -> list[str]:
    """WebDataset layout: shard-000000.tar, members <key>.txt and <key>.json."""
    os.makedirs(d, exist_ok=True)
    paths = []
    for k in range(0, len(ss), per_shard):
        p = os.path.join(d, f"shard-{k // per_shard:06d}.tar")
        # TODO: open tar for writing; for each sample add <key>.txt and <key>.json members (TarInfo with size, data from BytesIO)
        raise NotImplementedError("TODO(w21_training_layouts): open tar for writing; for each sample add <key>.txt and <key>.json members (TarInfo with size, data from BytesIO)")
        paths.append(p)
    return paths


def iter_tar(path):
    """Group consecutive members by basename into samples (WebDataset's rule)."""
    # TODO: stream members in order; when the basename changes, yield the finished sample
    raise NotImplementedError("TODO(w21_training_layouts): stream members in order; when the basename changes, yield the finished sample")


def write_parquet(ss, path, row_group: int = 1000):
    t = pa.table({"key": [s["key"] for s in ss], "text": [s["text"] for s in ss],
                  "meta": [json.dumps(s["meta"]) for s in ss]})
    pq.write_table(t, path, row_group_size=row_group)


def read_parquet(path):
    t = pq.read_table(path)
    return [{"key": k, "text": x, "meta": json.loads(m)} for k, x, m in
            zip(t["key"].to_pylist(), t["text"].to_pylist(), t["meta"].to_pylist())]


# ------------------------------------------------------------------ 2. resumable shuffle

class ShuffledStream:
    """Infinite, deterministic, resumable stream over tar shards."""

    def __init__(self, shards: list[str], seed: int = 0, buffer: int = 256):
        self.shards, self.seed, self.bufsize = list(shards), seed, buffer
        self.epoch, self.shard_pos, self.offset = 0, 0, 0
        self.buf: list[dict] = []
        self.rng = random.Random(seed)
        self._it, self._it_at = None, None   # open shard iterator (not part of the state)

    def _order(self) -> list[str]:
        o = list(self.shards)
        random.Random(self.seed * 1_000_003 + self.epoch).shuffle(o)
        return o

    def _fill(self):
        """Pull one sample from the shard cursor into the buffer."""
        while True:
            order = self._order()
            if self.shard_pos >= len(order):
                self.epoch, self.shard_pos, self.offset = self.epoch + 1, 0, 0
                continue
            if self._it is None or self._it_at != (self.epoch, self.shard_pos):
                self._it = iter_tar(order[self.shard_pos])
                self._it_at = (self.epoch, self.shard_pos)
                for _ in range(self.offset):   # re-opened after a restore: skip what was consumed
                    next(self._it)
            for s in self._it:
                self.offset += 1
                self.buf.append(s)
                if len(self.buf) >= self.bufsize:
                    return
            self.shard_pos, self.offset, self._it = self.shard_pos + 1, 0, None
            return

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        # TODO: top the buffer up to bufsize via _fill; pop a uniformly random element (swap with last, pop) using self.rng
        raise NotImplementedError("TODO(w21_training_layouts): top the buffer up to bufsize via _fill; pop a uniformly random element (swap with last, pop) using self.rng")

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "shard_pos": self.shard_pos, "offset": self.offset,
                "buf": [s["key"] for s in self.buf], "buf_full": list(self.buf), "rng": self.rng.getstate()}

    def load_state_dict(self, st: dict):
        self.epoch, self.shard_pos, self.offset = st["epoch"], st["shard_pos"], st["offset"]
        self.buf = list(st["buf_full"])
        self.rng.setstate(st["rng"])
        self._it = None


# ------------------------------------------------------------------ 3. mixtures

class Mixture:
    def __init__(self, sizes: dict[str, int], weights: dict[str, float], seed: int = 0):
        self.sizes, self.seed = sizes, seed
        tot = sum(weights.values())
        self.w = {k: v / tot for k, v in weights.items()}
        self.drawn = {k: 0 for k in sizes}

    def source_at(self, step: int) -> str:
        """Pure function of (seed, step): any worker can compute any step, restarts are free."""
        # TODO: u = Random(seed * 2**32 + step).random(); walk the cumulative weights (sorted by source name)
        raise NotImplementedError("TODO(w21_training_layouts): u = Random(seed * 2**32 + step).random(); walk the cumulative weights (sorted by source name)")

    def run(self, steps: int) -> dict:
        for t in range(steps):
            self.drawn[self.source_at(t)] += 1
        return {k: {"share": self.drawn[k] / steps, "epochs": self.drawn[k] / self.sizes[k]} for k in self.sizes}

    @staticmethod
    def capped_weights(sizes: dict[str, int], weights: dict[str, float], steps: int, max_epochs: float) -> dict[str, float]:
        """Lower weights of sources that would exceed max_epochs; give the mass to the others."""
        # TODO: iterate: cap w_k at max_epochs*size_k/steps for sources over the cap, renormalise the uncapped ones to absorb the rest
        raise NotImplementedError("TODO(w21_training_layouts): iterate: cap w_k at max_epochs*size_k/steps for sources over the cap, renormalise the uncapped ones to absorb the rest")


def report() -> str:
    import tempfile

    ss = samples(5000)
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for name, w, r in [
            ("loose files", lambda: write_files(ss, f"{d}/files"), lambda: read_files(f"{d}/files")),
            ("tar shards (1000/shard)", lambda: write_tar(ss, f"{d}/tar"),
             lambda: [s for p in sorted(pathlib.Path(f"{d}/tar").glob("*.tar")) for s in iter_tar(str(p))]),
            ("parquet", lambda: write_parquet(ss, f"{d}/x.parquet"), lambda: read_parquet(f"{d}/x.parquet")),
        ]:
            t0 = time.perf_counter(); w(); t1 = time.perf_counter()
            out = r(); t2 = time.perf_counter()
            assert [s["key"] for s in out] == [s["key"] for s in ss]
            path = pathlib.Path(d) / {"loose files": "files", "parquet": "x.parquet"}.get(name, "tar")
            nfiles = sum(1 for _ in path.rglob("*")) if path.is_dir() else 1
            rows.append({"layout": name, "files": nfiles, "write s": t1 - t0, "read s": t2 - t1})
    sizes = {"web": 1_000_000, "code": 100_000, "books": 8_000}
    weights = {"web": 0.6, "code": 0.25, "books": 0.15}
    steps = 400_000
    naive = Mixture(sizes, weights).run(steps)
    capped = Mixture(sizes, Mixture.capped_weights(sizes, weights, steps, 4.0)).run(steps)
    mix = [{"source": k, "weight": weights[k], "epochs (naive)": naive[k]["epochs"],
            "epochs (cap 4)": capped[k]["epochs"], "share (cap 4)": capped[k]["share"]} for k in sizes]
    return (f"5000 samples on local disk with a warm page cache ('files' = objects = GET requests on an object store; not measured here):\n" +
            table(rows, ["layout", "files", "write s", "read s"]) +
            f"\n\nMixture over {steps} draws:\n" +
            table(mix, ["source", "weight", "epochs (naive)", "epochs (cap 4)", "share (cap 4)"]))


if __name__ == "__main__":
    print(report())
