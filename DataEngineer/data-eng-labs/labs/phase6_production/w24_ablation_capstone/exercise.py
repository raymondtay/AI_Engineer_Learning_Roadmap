# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 24 capstone: does the pipeline actually help the model? Data ablations.

Roadmap: "Train the same tiny LM, same token budget, on the outputs of different
pipeline configurations; report held-out loss with seed variance. Write it up."

This is how FineWeb, DCLM and RefinedWeb justify every filter: train small
models on each variant at a fixed token budget and compare downstream metrics
(FineWeb used 1.82B-parameter models on 28B tokens per ablation; DCLM's
smallest competition scale is 412M parameters). Here: a character-level
transformer with ~0.25M parameters, 1.2M training characters per run, CPU.

  variants   raw | +heuristic filter | +filter +dedup (W22 pipeline) |
             +filter +dedup +"quality" (keep docs a small classifier trained on
             reference text likes - the DCLM/fastText idea, W09)
  metric     bits per character on held-out *clean* documents (stand-in for
             the downstream eval you care about)
  rigour     >= 2 training seeds per variant; report mean and spread. A
             difference smaller than the seed spread is not a result.
"""
from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from common.bench import table
from common.webgen import corpus, good_doc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 1))


def _w22():
    """Reuse YOUR Week 22 pipeline (exercise or solution, per LAB)."""
    lab = os.environ.get("LAB", "solution")
    p = ROOT / "labs/phase6_production/w22_incremental_pipeline" / f"{lab}.py"
    spec = importlib.util.spec_from_file_location(f"w22_{lab}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ data variants

def raw_pool(n: int = 1500, seed: int = 0) -> list[dict]:
    docs = corpus(n, seed=seed, mix={"good": 0.55, "spam": 0.15, "boilerplate": 0.12, "code": 0.08, "lorem": 0.10},
                  exact_dup=0.12, near_dup=0.12, langs={"en": 1.0})
    for i, d in enumerate(docs):
        d["id"] = i
    return docs


def reference_docs(n: int, seed: int) -> list[str]:
    import random
    rnd = random.Random(seed)
    return [good_doc(rnd, "en") for _ in range(n)]


def char_ngrams(text: str, n: int = 3) -> dict:
    t = f" {text.lower()} "
    out = {}
    for i in range(len(t) - n + 1):
        g = t[i : i + n]
        out[g] = out.get(g, 0) + 1
    return out


class NgramQuality:
    """Tiny multinomial naive Bayes: reference (clean) text vs random raw-pool text."""

    def fit(self, pos: list[str], neg: list[str]):
        self.lp = {}
        for label, docs in (("pos", pos), ("neg", neg)):
            c = {}
            for d in docs:
                for g, k in char_ngrams(d).items():
                    c[g] = c.get(g, 0) + k
            tot = sum(c.values()) + len(c) + 1
            self.lp[label] = ({g: math.log((k + 1) / tot) for g, k in c.items()}, math.log(1 / tot))
        return self

    def score(self, text: str) -> float:
        s = 0.0
        g = char_ngrams(text)
        n = sum(g.values()) or 1
        for gram, k in g.items():
            s += k * (self.lp["pos"][0].get(gram, self.lp["pos"][1]) - self.lp["neg"][0].get(gram, self.lp["neg"][1]))
        return s / n


def variants(pool: list[dict], seed: int = 0) -> dict[str, list[str]]:
    w22 = _w22()
    out = {"raw": [d["text"] for d in pool]}
    # TODO: 'filtered' = quality_ok heuristic; 'filtered+dedup' = W22 Pipeline.run_batch kept docs; 'filtered+dedup+classifier' = top 70% of those by NgramQuality fitted on 300 reference docs (seed+1000) vs 300 raw docs
    raise NotImplementedError("TODO(w24_ablation_capstone): 'filtered' = quality_ok heuristic; 'filtered+dedup' = W22 Pipeline.run_batch kept docs; 'filtered+dedup+classifier' = top 70% of those by NgramQuality fitted on 300 reference docs (seed+1000) vs 300 raw docs")
    return out


# ------------------------------------------------------------------ tiny LM

class CharLM(nn.Module):
    def __init__(self, vocab: int = 128, d: int = 96, layers: int = 2, heads: int = 4, ctx: int = 64):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, vocab)
        self.register_buffer("mask", torch.triu(torch.full((ctx, ctx), float("-inf")), 1))

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        return self.head(self.blocks(h, mask=self.mask[:T, :T], is_causal=True))


def encode(texts: list[str]) -> torch.Tensor:
    s = "\n\n".join(texts)
    return torch.tensor([min(ord(c), 127) for c in s], dtype=torch.long)


def train_eval(train_texts: list[str], val: torch.Tensor, steps: int = 300, batch: int = 64, seed: int = 0) -> float:
    """Fixed budget: steps*batch*ctx training characters, whatever the dataset size.
    Returns validation bits per character."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    data = encode(train_texts)
    model = CharLM().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    C = model.ctx
    # TODO: each step sample `batch` random windows of C+1 chars (generator g), next-char cross-entropy, AdamW step
    raise NotImplementedError("TODO(w24_ablation_capstone): each step sample `batch` random windows of C+1 chars (generator g), next-char cross-entropy, AdamW step")
    model.eval()
    with torch.no_grad():
        n = (len(val) - 1) // C * C
        xv = val[: n + 1]
        X = torch.stack([xv[i : i + C] for i in range(0, n, C)]).to(DEVICE)
        Y = torch.stack([xv[i + 1 : i + C + 1] for i in range(0, n, C)]).to(DEVICE)
        nll = F.cross_entropy(model(X).reshape(-1, 128), Y.reshape(-1)).item()
    return nll / math.log(2)


def ablate(steps: int = 300, seeds=(0, 1), n_pool: int = 1500) -> list[dict]:
    pool = raw_pool(n_pool)
    V = variants(pool)
    val = encode(reference_docs(60, seed=999))
    rows = []
    for name, texts in V.items():
        bpc = [train_eval(texts, val, steps=steps, seed=s) for s in seeds]
        rows.append({"variant": name, "docs": len(texts), "chars": sum(map(len, texts)),
                     "val bpc mean": float(np.mean(bpc)), "spread (max-min)": float(np.ptp(bpc))})
    return rows


def report() -> str:
    t0 = time.perf_counter()
    rows = ablate()
    return (f"device={DEVICE}; char transformer 2x96, 300 steps x 64 x 64 chars = 1.23M training chars per run, seeds 0 and 1\n"
            "held-out: 60 clean reference docs\n" +
            table(rows, ["variant", "docs", "chars", "val bpc mean", "spread (max-min)"]) +
            f"\n({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    print(report())
