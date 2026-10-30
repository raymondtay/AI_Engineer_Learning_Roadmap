# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 22: an incremental, idempotent cleaning pipeline.

Roadmap: "New crawl batches arrive daily. Process only what is new, dedup
against everything already kept, survive crashes and re-runs, and prove the
result equals a full recompute."

Common Crawl publishes a new crawl roughly monthly and much of each crawl
repeats URLs and content seen before; reprocessing the whole history every time
is the expensive mistake. The pieces:

  manifest       batch ids already committed -> re-running a batch is a no-op
  exact index    sha1 of normalised text of every kept document
  LSH index      MinHash band keys of every kept document (W07 parameters)
  commit         output file for the batch is written first (deterministic name,
                 so a retry overwrites it), then the new state is written to a
                 temp file and renamed over the old one. The rename is the commit
                 point: a crash before it means the batch is simply redone.
Equivalence test: incremental over batches 1..k == one full run over their
concatenation (same greedy first-seen order), with only new documents processed.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import pickle
import re
import sys
import time
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table
from common.webgen import corpus

BANDS, ROWS = 14, 8
K = BANDS * ROWS
PRIME = (1 << 61) - 1
_rng = np.random.default_rng(42)
A = _rng.integers(1, PRIME, K, dtype=np.uint64)
B = _rng.integers(0, PRIME, K, dtype=np.uint64)


class Crash(Exception):
    pass


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def minhash(text: str) -> np.ndarray:
    w = normalise(text).split()
    sh = [" ".join(w[i : i + 5]) for i in range(max(1, len(w) - 4))]
    h = np.array([zlib.crc32(s.encode()) for s in sh], dtype=np.uint64)
    return ((h[:, None] * A[None, :] + B[None, :]) % np.uint64(PRIME)).min(0)  # wraps like datasketch


def band_keys(sig: np.ndarray) -> list[str]:
    return [f"{b}:{hashlib.blake2b(sig[b * ROWS:(b + 1) * ROWS].tobytes(), digest_size=8).hexdigest()}" for b in range(BANDS)]


def quality_ok(text: str) -> bool:
    words = text.split()
    if len(words) < 50:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    return sum(l.rstrip().endswith((".", "!", "?")) for l in lines) / max(1, len(lines)) > 0.5


class Pipeline:
    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        (self.root / "out").mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.pkl"
        if self.state_path.exists():
            self.state = pickle.loads(self.state_path.read_bytes())
        else:
            self.state = {"manifest": [], "exact": set(), "lsh": {}}
        self.processed = 0

    def run_batch(self, batch_id: str, docs: list[dict], crash_after_output: bool = False) -> dict:
        if batch_id in self.state["manifest"]:
            return {"batch": batch_id, "skipped": True}
        # work on a copy: if we crash, the in-memory state must not have advanced either
        exact, lsh = set(self.state["exact"]), dict(self.state["lsh"])
        kept, stats = [], {"in": len(docs), "quality": 0, "exact dup": 0, "near dup": 0}
        for d in docs:
            self.processed += 1
            # TODO: quality filter; exact dedup on sha1(normalise(text)) against `exact`; near dedup: any band key already in `lsh` -> drop; else keep and register its hash and band keys (value = doc id)
            raise NotImplementedError("TODO(w22_incremental_pipeline): quality filter; exact dedup on sha1(normalise(text)) against `exact`; near dedup: any band key already in `lsh` -> drop; else keep and register its hash and band keys (value = doc id)")
        out = self.root / "out" / f"{batch_id}.parquet"
        pq.write_table(pa.table({"id": [d["id"] for d in kept], "text": [d["text"] for d in kept]}), out)
        if crash_after_output:
            raise Crash(batch_id)
        # TODO: new state = manifest + [batch_id], exact, lsh; pickle to a temp file in the same dir, os.replace onto state.pkl (the commit point), then adopt it in memory
        raise NotImplementedError("TODO(w22_incremental_pipeline): new state = manifest + [batch_id], exact, lsh; pickle to a temp file in the same dir, os.replace onto state.pkl (the commit point), then adopt it in memory")
        return {"batch": batch_id, "kept": len(kept), **stats}

    def kept_ids(self) -> list[int]:
        ids = []
        for b in self.state["manifest"]:   # only committed batches count
            ids += pq.read_table(self.root / "out" / f"{b}.parquet")["id"].to_pylist()
        return ids


def batches(n_batches: int = 5, per: int = 400, seed: int = 0):
    """Later batches re-crawl earlier pages (exact) and their edited copies (near)."""
    docs = corpus(n_batches * per, seed=seed, exact_dup=0.08, near_dup=0.12)
    for i, d in enumerate(docs):
        d["id"] = i
    return [docs[i * per:(i + 1) * per] for i in range(n_batches)]


def report() -> str:
    import tempfile

    bs = batches()
    rows = []
    with tempfile.TemporaryDirectory() as d:
        p = Pipeline(d)
        t_inc = 0.0
        for i, b in enumerate(bs):
            t0 = time.perf_counter()
            rows.append(p.run_batch(f"crawl-{i}", b))
            t_inc += time.perf_counter() - t0
        inc_ids, inc_work = p.kept_ids(), p.processed
        again = Pipeline(d).run_batch("crawl-2", bs[2])
    with tempfile.TemporaryDirectory() as d:
        full = Pipeline(d)
        work = 0
        t_full = 0.0
        for k in range(1, len(bs) + 1):   # "recompute everything every day"
            full = Pipeline(tempfile.mkdtemp(dir=d))
            t0 = time.perf_counter()
            full.run_batch("all", [x for b in bs[:k] for x in b])
            t_full += time.perf_counter() - t0
            work += full.processed
        full_ids = full.kept_ids()
    return (table(rows, ["batch", "in", "quality", "exact dup", "near dup", "kept"]) +
            f"\n\nincremental == full recompute: {inc_ids == full_ids} ({len(inc_ids)} docs)\n"
            f"docs processed over 5 days: incremental {inc_work} ({t_inc:.2f}s) vs daily full recompute {work} ({t_full:.2f}s)\n"
            f"re-running committed batch crawl-2: {again}")


if __name__ == "__main__":
    print(report())
