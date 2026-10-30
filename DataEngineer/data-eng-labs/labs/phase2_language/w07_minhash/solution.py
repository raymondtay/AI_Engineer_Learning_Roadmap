"""Week 7: fuzzy deduplication with MinHash LSH.

Roadmap: "MinHash LSH from scratch: shingles, signatures, bands, the S-curve;
then the datatrove pipeline. Precision/recall against exact Jaccard on a
labelled sample."

  shingles    set of word n-grams of a document (FineWeb: 5-grams)
  signature   for k hash functions, the minimum hash over the shingles;
              P[min-hashes equal] = Jaccard similarity
  banding     split k = b * r into b bands of r rows; two documents become a
              candidate pair if ANY band matches exactly
  S-curve     P(candidate | Jaccard s) = 1 - (1 - s^r)^b
              FineWeb uses b=14, r=8 (112 hashes) and reports 56% / 77% / 92% /
              98.8% at s = 0.70 / 0.75 / 0.80 / 0.85 - test_lab checks your
              formula against those numbers.
  clusters    union-find over candidate pairs; keep one document per cluster

Exact all-pairs Jaccard is O(n^2); LSH is roughly linear. The lab measures
both, and the precision/recall LSH gives up for that speed.
"""
from __future__ import annotations

import itertools
import pathlib
import random
import sys
import time
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np

from common.bench import prf, table
from common.webgen import good_doc, perturb

PRIME = (1 << 61) - 1


def shingles(text: str, n: int = 5) -> set[int]:
    w = text.lower().split()
    # >>> SOLUTION: crc32 of each space-joined word n-gram (whole text if shorter than n)
    if len(w) < n:
        return {zlib.crc32(" ".join(w).encode())}
    return {zlib.crc32(" ".join(w[i : i + n]).encode()) for i in range(len(w) - n + 1)}
    # <<< SOLUTION


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


class MinHasher:
    """k hash functions h_i(x) = (a_i * x + b_i) mod P, with a_i, b_i drawn over
    the full range [0, P) and uint64 arithmetic allowed to wrap (as datasketch does).

    A trap measured while building this lab: drawing a_i, b_i below 2^32 only
    (so a*x never overflows) looks safer but is far from min-wise independent -
    on a pair with true Jaccard 0.64 it estimated 0.76 even with 2,048 hashes.
    The full-range version is unbiased within sampling noise."""

    def __init__(self, k: int = 112, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, PRIME, size=k, dtype=np.uint64)
        self.b = rng.integers(0, PRIME, size=k, dtype=np.uint64)
        self.k = k

    def signature(self, sh: set[int]) -> np.ndarray:
        x = np.fromiter(sh, dtype=np.uint64, count=len(sh))
        # >>> SOLUTION: h = (a * x + b) % P for every (hash fn, shingle) via broadcasting (uint64 wraps); min over shingles
        h = (self.a[:, None] * x[None, :] + self.b[:, None]) % np.uint64(PRIME)
        return h.min(axis=1)
        # <<< SOLUTION


def s_curve(s: float, bands: int, rows: int) -> float:
    # >>> SOLUTION: 1 - (1 - s**rows) ** bands
    return 1 - (1 - s**rows) ** bands
    # <<< SOLUTION


def lsh_pairs(sigs: np.ndarray, bands: int, rows: int) -> set[tuple[int, int]]:
    """Candidate pairs (i < j) sharing at least one identical band."""
    assert sigs.shape[1] == bands * rows
    pairs: set[tuple[int, int]] = set()
    # >>> SOLUTION: for each band, bucket documents by the bytes of sigs[:, band slice]; every pair inside a bucket is a candidate
    for b in range(bands):
        buckets: dict[bytes, list[int]] = {}
        for i, row in enumerate(sigs[:, b * rows : (b + 1) * rows]):
            buckets.setdefault(row.tobytes(), []).append(i)
        for ids in buckets.values():
            if len(ids) > 1:
                pairs.update(itertools.combinations(ids, 2))
    # <<< SOLUTION
    return pairs


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dedup(texts: list[str], bands: int = 14, rows: int = 8, n: int = 5) -> list[int]:
    """Indices to KEEP (lowest index of each near-duplicate cluster)."""
    mh = MinHasher(bands * rows)
    sigs = np.stack([mh.signature(shingles(t, n)) for t in texts])
    uf = UnionFind(len(texts))
    for i, j in lsh_pairs(sigs, bands, rows):
        uf.union(i, j)
    return sorted({uf.find(i) for i in range(len(texts))})


def make_dataset(n_src: int = 150, seed: int = 0) -> list[str]:
    """Sources plus near-duplicates edited at rates spanning the threshold region."""
    rnd = random.Random(seed)
    texts = [good_doc(rnd) for _ in range(n_src)]
    for i in range(n_src):
        for rate in rnd.sample([0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3], 2):
            texts.append(perturb(rnd, texts[i], rate))
    return texts


def evaluate(texts: list[str], threshold: float = 0.75, configs=((14, 8), (20, 5), (9, 13))) -> list[dict]:
    sh = [shingles(t) for t in texts]
    t0 = time.perf_counter()
    truth = {(i, j) for i, j in itertools.combinations(range(len(texts)), 2) if jaccard(sh[i], sh[j]) >= threshold}
    t_exact = time.perf_counter() - t0
    rows = []
    for b, r in configs:
        mh = MinHasher(b * r)
        t0 = time.perf_counter()
        sigs = np.stack([mh.signature(s) for s in sh])
        pred = lsh_pairs(sigs, b, r)
        t_lsh = time.perf_counter() - t0
        m = prf(pred, truth)
        rows.append({"bands x rows": f"{b}x{r}", "P(cand|s=thr)": s_curve(threshold, b, r), "precision": m["precision"],
                     "recall": m["recall"], "LSH s": t_lsh, "exact s": t_exact})
    return rows


def _demo():
    for s in (0.7, 0.75, 0.8, 0.85):
        print(f"S-curve b=14 r=8 at s={s}: {s_curve(s, 14, 8):.3f}")
    print(table(evaluate(make_dataset()), ["bands x rows", "P(cand|s=thr)", "precision", "recall", "LSH s", "exact s"]))


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
