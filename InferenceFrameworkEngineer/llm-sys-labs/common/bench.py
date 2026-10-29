"""Timing that is honest on both CPU and GPU.

GPU work is asynchronous, so a wall-clock timer without synchronisation
measures kernel *launch* time. `bench` synchronises around every sample and
reports the median, which is robust to the occasional slow iteration.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn: Callable[[], object], warmup: int = 3, iters: int = 10) -> float:
    """Median seconds per call."""
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def table(rows: list[dict], cols: list[str]) -> str:
    """Render a small markdown table for lab reports."""
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
