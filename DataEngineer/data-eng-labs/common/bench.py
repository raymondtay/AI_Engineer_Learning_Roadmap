"""Timing and reporting helpers (no GPU in this track)."""
from __future__ import annotations

import statistics
import time
from typing import Callable


def bench(fn: Callable[[], object], warmup: int = 1, iters: int = 5) -> float:
    """Median seconds per call."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def prf(pred: set, truth: set) -> dict:
    """Precision, recall, F1 of a predicted set against a true set."""
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(truth) if truth else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}
