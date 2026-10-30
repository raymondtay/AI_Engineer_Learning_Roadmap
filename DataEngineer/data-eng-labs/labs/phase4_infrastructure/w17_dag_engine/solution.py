"""Week 17: a DAG engine with content-addressed caching and a Rust hot path.

Roadmap: "A mini Airflow/Dagster: tasks, dependencies, retries, caching keyed by
code + inputs, failure isolation. Profile the pipeline; move the hot loop to Rust."

Engine
  Task(name, fn, deps, params, version)
  run(): topological order, independent tasks in parallel (thread pool), each
         task retried up to `retries` times, a failed task marks everything
         downstream 'skipped' while unrelated branches finish.
  cache: key = sha256(name, version, params, keys of deps). Same key -> load the
         stored result instead of running. Change one param -> only that task and
         its descendants rerun. This is how Dagster/Bazel/Nix-style systems avoid
         recomputing a 100 TB pipeline after a one-line filter change.

Hot path
  Profiling the Phase 2 pipeline shows MinHash dominating. `rust_minhash/` is a
  std-only Rust cdylib with a C ABI, called through ctypes, that must agree bit
  for bit with `minhash_py`. Build: cargo build --release (in rust_minhash/).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import pathlib
import pickle
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table

HERE = pathlib.Path(__file__).resolve().parent
MASK = (1 << 64) - 1


# ------------------------------------------------------------------ engine

@dataclass
class Task:
    name: str
    fn: Callable                 # fn(*dep_results, **params)
    deps: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    version: str = "1"           # bump when the code changes
    retries: int = 0


class DAG:
    def __init__(self, cache_dir: str, workers: int = 4):
        self.tasks: dict[str, Task] = {}
        self.cache = pathlib.Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.log: list[tuple[str, str]] = []   # (task, event): ran | cached | failed | skipped | retry
        self._lock = threading.Lock()

    def add(self, task: Task):
        self.tasks[task.name] = task
        return task

    def topo(self) -> list[str]:
        """Kahn's algorithm; raises ValueError on a cycle or unknown dependency."""
        # >>> SOLUTION: count in-degrees, repeatedly emit tasks with in-degree 0 (sorted for determinism); leftover tasks mean a cycle
        for t in self.tasks.values():
            for d in t.deps:
                if d not in self.tasks:
                    raise ValueError(f"{t.name} depends on unknown {d}")
        indeg = {n: len(t.deps) for n, t in self.tasks.items()}
        children = {n: [] for n in self.tasks}
        for n, t in self.tasks.items():
            for d in t.deps:
                children[d].append(n)
        ready = sorted(n for n, k in indeg.items() if k == 0)
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
            ready.sort()
        if len(order) != len(self.tasks):
            raise ValueError("cycle among " + ", ".join(sorted(set(self.tasks) - set(order))))
        return order
        # <<< SOLUTION

    def key(self, name: str, keys: dict[str, str]) -> str:
        t = self.tasks[name]
        # >>> SOLUTION: sha256 over json of [name, version, sorted params, [keys[d] for d in deps]]
        payload = json.dumps([name, t.version, sorted(t.params.items()), [keys[d] for d in t.deps]], default=str)
        return hashlib.sha256(payload.encode()).hexdigest()
        # <<< SOLUTION

    def _event(self, name, ev):
        with self._lock:
            self.log.append((name, ev))

    def _execute(self, name: str, inputs: list, key: str):
        t = self.tasks[name]
        path = self.cache / f"{key}.pkl"
        if path.exists():
            self._event(name, "cached")
            return pickle.loads(path.read_bytes())
        for attempt in range(t.retries + 1):
            try:
                out = t.fn(*inputs, **t.params)
                break
            except Exception:
                if attempt == t.retries:
                    raise
                self._event(name, "retry")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(out))
        tmp.replace(path)          # atomic: a crash never leaves a half-written cache entry
        self._event(name, "ran")
        return out

    def run(self) -> dict:
        """Returns {name: result} for tasks that succeeded."""
        order = self.topo()
        self.log = []
        keys: dict[str, str] = {n: self.key(n, {}) if not self.tasks[n].deps else "" for n in order}
        for n in order:                       # keys depend only on upstream keys, not results
            keys[n] = self.key(n, keys)
        results, state = {}, {n: "pending" for n in order}
        running = {}
        with ThreadPoolExecutor(self.workers) as ex:
            while True:
                # >>> SOLUTION: mark pending tasks with any failed/skipped dep as skipped (log it); submit pending tasks whose deps are all done; stop when nothing is running and nothing can be submitted; wait FIRST_COMPLETED and record done/failed
                progressed = True
                while progressed:
                    progressed = False
                    for n in order:
                        if state[n] != "pending":
                            continue
                        deps = self.tasks[n].deps
                        if any(state[d] in ("failed", "skipped") for d in deps):
                            state[n] = "skipped"
                            self._event(n, "skipped")
                            progressed = True
                        elif all(state[d] == "done" for d in deps):
                            state[n] = "running"
                            running[ex.submit(self._execute, n, [results[d] for d in deps], keys[n])] = n
                if not running:
                    break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for f in finished:
                    n = running.pop(f)
                    try:
                        results[n] = f.result()
                        state[n] = "done"
                    except Exception:
                        state[n] = "failed"
                        self._event(n, "failed")
                # <<< SOLUTION
        self.state = state
        return results

    def events(self, ev: str) -> list[str]:
        return sorted(n for n, e in self.log if e == ev)


# ------------------------------------------------------------------ hot path

def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def _fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for c in data:
        h = ((h ^ c) * 0x100000001B3) & MASK
    return h


def minhash_py(doc: str, k: int = 64, seed: int = 0) -> list[int]:
    """Reference implementation. Slow on purpose: this is what the profiler flags."""
    words = doc.split()
    a = [_splitmix64((seed + 2 * i) & MASK) | 1 for i in range(k)]
    b = [_splitmix64((seed + 2 * i + 1) & MASK) for i in range(k)]
    shingles = [" ".join(words)] if len(words) < 5 else [" ".join(words[s : s + 5]) for s in range(len(words) - 4)]
    sig = [MASK] * k
    for sh in shingles:
        h = _fnv1a(sh.encode())
        for i in range(k):
            v = (a[i] * h + b[i]) & MASK
            if v < sig[i]:
                sig[i] = v
    return sig


def build_rust() -> pathlib.Path | None:
    so = HERE / "rust_minhash" / "target" / "release" / "librust_minhash.so"
    if not so.exists():
        try:
            subprocess.run(["cargo", "build", "--release", "-q"], cwd=HERE / "rust_minhash", check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
    return so


def minhash_rust(docs: list[str], k: int = 64, seed: int = 0) -> np.ndarray:
    so = build_rust()
    if so is None:
        raise RuntimeError("cargo not available")
    lib = ctypes.CDLL(str(so))
    # >>> SOLUTION: pack docs into one bytes buffer + uint64 offsets (n+1), allocate out uint64[n, k], declare argtypes and call lib.minhash_batch
    enc = [d.encode() for d in docs]
    buf = b"".join(enc)
    offsets = np.zeros(len(docs) + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum([len(e) for e in enc])
    out = np.empty((len(docs), k), dtype=np.uint64)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    lib.minhash_batch.argtypes = [ctypes.c_char_p, u64p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, u64p]
    lib.minhash_batch.restype = None
    lib.minhash_batch(buf, offsets.ctypes.data_as(u64p), len(docs), k, seed, out.ctypes.data_as(u64p))
    return out
    # <<< SOLUTION


# ------------------------------------------------------------------ demo pipeline

def pipeline(cache_dir: str, min_words: int = 50, use_rust: bool = False, n_docs: int = 400) -> DAG:
    from common.webgen import corpus

    dag = DAG(cache_dir)
    dag.add(Task("load", lambda n: [x["text"] for x in corpus(n, seed=0)], params={"n": n_docs}))
    dag.add(Task("filter", lambda docs, min_words: [d for d in docs if len(d.split()) >= min_words],
                 deps=["load"], params={"min_words": min_words}))
    if use_rust:
        dag.add(Task("minhash", lambda docs: minhash_rust(docs).tolist(), deps=["filter"], version="rust-1"))
    else:
        dag.add(Task("minhash", lambda docs: [minhash_py(d) for d in docs], deps=["filter"], version="py-1"))
    dag.add(Task("stats", lambda docs: {"docs": len(docs), "words": sum(len(d.split()) for d in docs)}, deps=["load"]))
    dag.add(Task("report", lambda sigs, st: {"sigs": len(sigs), **st}, deps=["minhash", "stats"]))
    return dag


def report() -> str:
    import tempfile

    from common.webgen import corpus

    rows = []
    with tempfile.TemporaryDirectory() as d:
        dag = pipeline(d)
        t0 = time.perf_counter(); dag.run(); t1 = time.perf_counter()
        rows.append({"run": "cold", "ran": len(dag.events("ran")), "cached": len(dag.events("cached")), "s": t1 - t0})
        t0 = time.perf_counter(); dag.run(); t1 = time.perf_counter()
        rows.append({"run": "warm (nothing changed)", "ran": len(dag.events("ran")), "cached": len(dag.events("cached")), "s": t1 - t0})
        dag = pipeline(d, min_words=60)
        t0 = time.perf_counter(); dag.run(); t1 = time.perf_counter()
        rows.append({"run": "filter param changed", "ran": len(dag.events("ran")), "cached": len(dag.events("cached")), "s": t1 - t0})
    docs = [x["text"] for x in corpus(1000, seed=1)]
    t0 = time.perf_counter(); py = [minhash_py(x) for x in docs[:200]]; tp = (time.perf_counter() - t0) / 200
    hot = []
    if build_rust():
        minhash_rust(docs[:10])
        t0 = time.perf_counter(); rs = minhash_rust(docs); tr = (time.perf_counter() - t0) / len(docs)
        assert rs[:200].tolist() == py
        hot = [{"impl": "python", "ms/doc": tp * 1e3}, {"impl": "rust (ctypes)", "ms/doc": tr * 1e3},
               {"impl": "speed-up", "ms/doc": tp / tr}]
    return (table(rows, ["run", "ran", "cached", "s"]) + "\n\nMinHash k=64, 5-gram shingles:\n" +
            table(hot, ["impl", "ms/doc"]))


if __name__ == "__main__":
    print(report())
