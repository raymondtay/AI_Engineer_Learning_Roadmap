"""Week 8: exact, substring and semantic deduplication.

Roadmap: "URL and exact-document hashing, exact-substring dedup with a suffix
array, SemDeDup-style embedding clusters."

Three duplicates MinHash handles badly, one method each:
  1. exact copies across URLs        -> hash of normalised text (cheap; do it first)
  2. a shared span inside otherwise   -> suffix array: find substrings of >= L bytes
     different documents (newsletter     occurring in 2+ documents, cut them from all
     blurbs, licence text)               but the first (Lee et al. 2022: one 61-word
                                         sentence repeated 60,000+ times in C4)
  3. same content, different wording  -> embeddings: cluster, then drop near-identical
     or order                            vectors within a cluster (SemDeDup)

The embedding here is a hashed bag of words through a random projection, so the
lab runs offline. In production use a sentence-embedding model; the clustering
and per-cluster dedup code does not change.
"""
from __future__ import annotations

import hashlib
import pathlib
import random
import re
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np

from common.bench import table
from common.webgen import good_doc


# ------------------------------------------------------------------ 1. exact

def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def exact_dedup(texts: list[str]) -> list[int]:
    """Indices to keep: first occurrence of each normalised text."""
    # >>> SOLUTION: sha1 of normalise(text); keep index when the digest is new
    seen, keep = set(), []
    for i, t in enumerate(texts):
        h = hashlib.sha1(normalise(t).encode()).digest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep
    # <<< SOLUTION


# ------------------------------------------------------------------ 2. suffix array

def suffix_array(data: bytes) -> list[int]:
    """Prefix-doubling suffix array, O(n log^2 n): fine for a lab, use a C++/Rust
    implementation (as Lee et al. did) for real corpora."""
    n = len(data)
    rank = list(data)
    sa = list(range(n))
    k = 1
    while True:
        key = lambda i: (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)
        new = [0] * n
        for j in range(1, n):
            new[sa[j]] = new[sa[j - 1]] + (key(sa[j]) != key(sa[j - 1]))
        rank = new
        if rank[sa[-1]] == n - 1:
            return sa
        k *= 2


def lcp(data: bytes, i: int, j: int, cap: int) -> int:
    n, k = len(data), 0
    while i + k < n and j + k < n and k < cap and data[i + k] == data[j + k]:
        k += 1
    return k


def repeated_spans(texts: list[str], min_len: int = 100) -> list[list[tuple[int, int]]]:
    """For each document, byte ranges that also occur (>= min_len bytes) in an EARLIER document."""
    sep = b"\x00"
    blobs = [t.encode() for t in texts]
    data = sep.join(blobs)
    starts, off = [], 0
    for b in blobs:
        starts.append(off)
        off += len(b) + 1
    doc_of = np.searchsorted(np.array(starts), np.arange(len(data)), side="right") - 1
    sa = suffix_array(data)
    spans: list[list[tuple[int, int]]] = [[] for _ in texts]
    # >>> SOLUTION: group consecutive suffixes whose adjacent LCP >= min_len into runs; in each run spanning 2+ docs, cut every member except those in the earliest doc (length = its larger neighbour LCP)
    # Comparing only adjacent PAIRS and cutting the later doc of each pair is a
    # trap: a copy whose two neighbours both come from later docs is never cut.
    lcps = [lcp(data, a, b, 10_000) for a, b in zip(sa, sa[1:])]
    i = 0
    while i < len(lcps):
        if lcps[i] < min_len:
            i += 1
            continue
        j = i
        while j < len(lcps) and lcps[j] >= min_len:
            j += 1
        members = sa[i : j + 1]                      # suffixes i..j form one run
        docs = [int(doc_of[p]) for p in members]
        first = min(docs)
        if len(set(docs)) > 1:
            for k, (p, d) in enumerate(zip(members, docs)):
                if d == first:
                    continue
                L = max(lcps[i + k - 1] if k > 0 else 0, lcps[i + k] if i + k < len(lcps) else 0)
                s = p - starts[d]
                spans[d].append((s, min(s + L, len(blobs[d]))))
        i = j + 1
    # <<< SOLUTION
    return spans


def remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    b = text.encode()
    keep = bytearray()
    cut = sorted(spans)
    pos = 0
    for s, e in cut:
        if s > pos:
            keep += b[pos:s]
        pos = max(pos, e)
    keep += b[pos:]
    return keep.decode(errors="ignore")


# ------------------------------------------------------------------ 3. semantic

def embed(texts: list[str], dim: int = 64, buckets: int = 4096, seed: int = 0) -> np.ndarray:
    """Hashed bag of words -> TF-IDF -> random projection -> unit vectors."""
    X = np.zeros((len(texts), buckets))
    for i, t in enumerate(texts):
        for w in re.findall(r"\w+", t.lower()):
            X[i, zlib.crc32(w.encode()) % buckets] += 1
    df = (X > 0).sum(0)
    X *= np.log((1 + len(texts)) / (1 + df)) + 1
    R = np.random.default_rng(seed).standard_normal((buckets, dim))
    E = X @ R
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-9)


def kmeans(E: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = E[rng.choice(len(E), k, replace=False)]
    for _ in range(iters):
        lab = (E @ C.T).argmax(1)
        for j in range(k):
            if (lab == j).any():
                c = E[lab == j].mean(0)
                C[j] = c / max(np.linalg.norm(c), 1e-9)
    return (E @ C.T).argmax(1)


def semdedup(E: np.ndarray, k: int = 10, eps: float = 0.05) -> list[int]:
    """Within each cluster, drop any point with cosine >= 1 - eps to an earlier kept point."""
    labels = kmeans(E, k)
    keep = []
    # >>> SOLUTION: for each cluster, iterate its members in index order; keep a member unless its max cosine to already-kept members of the cluster >= 1 - eps
    for c in np.unique(labels):
        kept_c: list[int] = []
        for i in np.flatnonzero(labels == c):
            if not kept_c or float((E[kept_c] @ E[i]).max()) < 1 - eps:
                kept_c.append(int(i))
        keep += kept_c
    # <<< SOLUTION
    return sorted(keep)


# ------------------------------------------------------------------ data + report

NEWSLETTER = ("Subscribe to our weekly newsletter to receive the latest stories, exclusive interviews and "
              "special offers directly in your inbox every Friday morning, and unsubscribe at any time.")


def reorder(rnd, text: str) -> str:
    """Same words, different order inside each sentence: meaning-level duplicate, 5-gram shingles destroyed."""
    out = []
    for s in re.split(r"(?<=[.?!])\s+", text):
        w = s.rstrip(".?!").split()
        rnd.shuffle(w)
        out.append(" ".join(w) + ".")
    return " ".join(out)


def make_dataset(n: int = 120, seed: int = 0) -> dict:
    rnd = random.Random(seed)
    src = [good_doc(rnd) for _ in range(n)]
    exact = [src[i] for i in range(0, n, 10)]
    semantic = [reorder(rnd, src[i]) for i in range(5, n, 10)]
    with_blurb = [src[i] + "\n" + NEWSLETTER for i in range(n)]
    return {"src": src, "exact": exact, "semantic": semantic, "with_blurb": with_blurb}


def summary(seed: int = 0) -> list[dict]:
    d = make_dataset(seed=seed)
    texts = d["src"] + d["exact"] + d["semantic"]
    n = len(d["src"])
    keep_exact = exact_dedup(texts)
    E = embed(texts)
    keep_sem = semdedup(E)
    spans = repeated_spans(d["with_blurb"][:40])
    cleaned = [remove_spans(t, s) for t, s in zip(d["with_blurb"][:40], spans)]
    return [
        {"method": "exact hash", "removed exact copies": sum(i >= n and i < n + len(d["exact"]) for i in set(range(len(texts))) - set(keep_exact)),
         "removed reordered": sum(i >= n + len(d["exact"]) for i in set(range(len(texts))) - set(keep_exact)),
         "removed originals": sum(i < n for i in set(range(len(texts))) - set(keep_exact))},
        {"method": "semantic (embeddings)", "removed exact copies": sum(n <= i < n + len(d["exact"]) for i in set(range(len(texts))) - set(keep_sem)),
         "removed reordered": sum(i >= n + len(d["exact"]) for i in set(range(len(texts))) - set(keep_sem)),
         "removed originals": sum(i < n for i in set(range(len(texts))) - set(keep_sem))},
        {"method": "suffix array (newsletter blurb)", "removed exact copies": "-", "removed reordered": "-",
         "removed originals": f"blurb kept in doc 0: {NEWSLETTER[:20] in cleaned[0]}; in others: {sum(NEWSLETTER[:20] in c for c in cleaned[1:])}"},
    ]


def _demo():
    d = make_dataset()
    print(f"{len(d['src'])} originals, {len(d['exact'])} exact copies, {len(d['semantic'])} reordered copies")
    print(table(summary(), ["method", "removed exact copies", "removed reordered", "removed originals"]))


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
