"""Week 4: parallel processing - map, hash shuffle, reduce, external sort.

Roadmap: "Map, hash-partitioned shuffle, external sort, all with
multiprocessing; then the same job in Ray or Spark. Measure scaling with cores."

The job: per-host document counts and word counts across many shards, the
shape of almost every corpus statistic. Structure (what Spark does for you):

  map      each worker reads one shard, emits (key, value) pairs
  shuffle  each pair goes to partition stable_hash(key) % R, spilled to disk
  reduce   each worker merges one partition

The trap this lab is built around: Python's built-in hash() of a str is
randomised per process (PYTHONHASHSEED). Use it to partition and the same key
lands in different partitions on different workers, so reducers see partial
counts. Use a stable hash.

External sort: sort data bigger than memory by sorting chunks, spilling them,
and k-way merging the runs - how dedup and joins work at scale.
"""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
import os
import pathlib
import pickle
import sys
import tempfile
import time
from multiprocessing import get_context

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import pyarrow.parquet as pq

from common.bench import table
from common.data import docs_table


def stable_hash(key: str) -> int:
    """Same value in every process and every run."""
    # >>> SOLUTION: first 8 bytes of blake2b(key) as an unsigned int
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    # <<< SOLUTION


def partition_of(key: str, n_parts: int) -> int:
    return stable_hash(key) % n_parts


def map_shard(args) -> list[str]:
    """Read one shard, count (host, words) locally, spill each partition to its own file."""
    shard, n_parts, spill_dir, task = args
    t = pq.read_table(shard, columns=["host", "text"])
    local: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for host, text in zip(t["host"].to_pylist(), t["text"].to_pylist()):
        c = local[host]
        c[0] += 1
        c[1] += len(text.split())
    paths = []
    # >>> SOLUTION: bucket local items by partition_of(host, n_parts); write each bucket to spill_dir/part-{p}-task-{task}.pkl; return paths
    buckets: dict[int, list] = collections.defaultdict(list)
    for host, (d, w) in local.items():
        buckets[partition_of(host, n_parts)].append((host, d, w))
    for p, items in buckets.items():
        path = os.path.join(spill_dir, f"part-{p}-task-{task}.pkl")
        with open(path, "wb") as f:
            pickle.dump(items, f)
        paths.append(path)
    # <<< SOLUTION
    return paths


def reduce_partition(args) -> dict[str, tuple[int, int]]:
    p, spill_dir = args
    out: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    # >>> SOLUTION: load every part-{p}-task-*.pkl and sum (docs, words) per host
    for f in pathlib.Path(spill_dir).glob(f"part-{p}-task-*.pkl"):
        with open(f, "rb") as fh:
            for host, d, w in pickle.load(fh):
                out[host][0] += d
                out[host][1] += w
    # <<< SOLUTION
    return {k: tuple(v) for k, v in out.items()}


def host_stats(shards: list[str], workers: int, n_parts: int = 8) -> dict[str, tuple[int, int]]:
    # fork: workers inherit the already-imported lab module (spawn would have to
    # re-import it by name, which fails for labs loaded from a file path).
    with tempfile.TemporaryDirectory() as spill, get_context("fork").Pool(workers) as pool:
        pool.map(map_shard, [(s, n_parts, spill, i) for i, s in enumerate(shards)])
        parts = pool.map(reduce_partition, [(p, spill) for p in range(n_parts)])
    result: dict[str, tuple[int, int]] = {}
    for part in parts:
        for k, v in part.items():
            assert k not in result, f"key {k} reduced in two partitions: the shuffle is not stable"
            result[k] = v
    return result


def external_sort(items, chunk: int, spill_dir: str):
    """Yield items in sorted order using at most `chunk` items in memory per run."""
    runs = []
    buf = []
    # >>> SOLUTION: fill buf to `chunk`, sort, spill to a JSONL run file; at end spill the rest; then heapq.merge readers of all runs
    def spill():
        path = os.path.join(spill_dir, f"run-{len(runs)}.jsonl")
        with open(path, "w") as f:
            for x in sorted(buf):
                f.write(json.dumps(x) + "\n")
        runs.append(path)
        buf.clear()

    for x in items:
        buf.append(x)
        if len(buf) >= chunk:
            spill()
    if buf:
        spill()

    def read(p):
        with open(p) as f:
            for line in f:
                yield json.loads(line)

    yield from heapq.merge(*(read(p) for p in runs))
    # <<< SOLUTION


def make_shards(n_shards: int, docs_per: int, d: str) -> list[str]:
    paths = []
    for i in range(n_shards):
        p = f"{d}/shard-{i:03d}.parquet"
        pq.write_table(docs_table(docs_per, seed=i), p)
        paths.append(p)
    return paths


def scaling(n_shards: int = 8, docs_per: int = 5000) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        shards = make_shards(n_shards, docs_per, d)
        rows = []
        for w in (1, 2, 4):
            t0 = time.perf_counter()
            host_stats(shards, w)
            rows.append({"workers": w, "seconds": time.perf_counter() - t0})
        for r in rows:
            r["speedup"] = rows[0]["seconds"] / r["seconds"]
    return rows


def _demo():
    print(f"cores: {os.cpu_count()}")
    print(table(scaling(), ["workers", "seconds", "speedup"]))


def report() -> str:
    """The gate measurement as text (used by tools/gates.py)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _demo()
    return buf.getvalue()


if __name__ == "__main__":
    print(report())
