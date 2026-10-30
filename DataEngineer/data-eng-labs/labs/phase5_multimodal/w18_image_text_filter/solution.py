"""Week 18: image-text filtering with a contrastive model (CLIP score).

Roadmap: "Train a tiny CLIP on synthetic pairs; filter a noisy pool by CLIP
score; compare against caption heuristics. Read DataComp."

LAION-5B kept pairs whose OpenAI-CLIP ViT-B/32 cosine similarity was >= 0.28
(English) / 0.26 (other languages) (Schuhmann et al. 2022, sec. 3). DataComp
(Gadre et al. 2023) then showed that *which* pairs you keep matters as much as
the model: CLIP-score filtering beat no filtering at every scale, and the best
baselines kept roughly the top 30% by score.

You will:
  1. train a tiny CLIP (CNN image tower, bag-of-words text tower, InfoNCE loss)
     on a clean pool
  2. score a noisy pool (wrong captions + junk alt-text) and keep the top q%
  3. compare with a text-only heuristic (does the caption contain a colour and
     a shape word?) - which catches junk but not *wrong* captions
Runs on CPU in ~30 s; uses CUDA if present.
"""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table
from common.shapes import COLORS, SHAPES, pairs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Vocab:
    def __init__(self, captions: list[str]):
        words = sorted({w for c in captions for w in re.findall(r"[a-z]+", c.lower())})
        self.idx = {w: i + 1 for i, w in enumerate(words)}   # 0 = unknown

    def encode(self, caps: list[str], max_len: int = 8) -> torch.Tensor:
        out = torch.zeros(len(caps), max_len, dtype=torch.long)
        for i, c in enumerate(caps):
            ids = [self.idx.get(w, 0) for w in re.findall(r"[a-z]+", c.lower())][:max_len]
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out


class TinyCLIP(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 64):
        super().__init__()
        self.img = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, dim))
        self.emb = nn.Embedding(vocab_size + 1, dim, padding_idx=0)
        self.txt = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.img(x), dim=-1)

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        # >>> SOLUTION: mean of token embeddings over non-padding positions, then the MLP, then L2-normalise
        m = (ids > 0).float().unsqueeze(-1)
        e = (self.emb(ids) * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.txt(e), dim=-1)
        # <<< SOLUTION


def to_tensor(imgs: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255)


def clip_loss(img_emb, txt_emb, logit_scale) -> torch.Tensor:
    """Symmetric InfoNCE (Radford et al. 2021, fig. 3)."""
    # >>> SOLUTION: logits = scale * img @ txt.T; targets = arange(B); mean of CE over rows and over columns
    logits = logit_scale.exp().clamp(max=100) * img_emb @ txt_emb.t()
    y = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y)) / 2
    # <<< SOLUTION


def train(n: int = 4000, epochs: int = 6, seed: int = 0, batch: int = 128):
    torch.manual_seed(seed)
    imgs, caps, _ = pairs(n, seed=seed)
    vocab = Vocab(caps + [f"{c} {s}" for c in COLORS for s in SHAPES])
    model = TinyCLIP(len(vocab.idx)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    X, T = to_tensor(imgs), vocab.encode(caps)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            j = perm[i : i + batch]
            loss = clip_loss(model.encode_image(X[j].to(DEVICE)), model.encode_text(T[j].to(DEVICE)), model.logit_scale)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, vocab


@torch.no_grad()
def clip_scores(model, vocab, imgs, caps) -> np.ndarray:
    """Cosine similarity of each image with ITS OWN caption."""
    # >>> SOLUTION: encode both in chunks, row-wise dot product
    out = []
    for i in range(0, len(caps), 512):
        a = model.encode_image(to_tensor(imgs[i : i + 512]).to(DEVICE))
        b = model.encode_text(vocab.encode(caps[i : i + 512]).to(DEVICE))
        out.append((a * b).sum(-1).cpu())
    return torch.cat(out).numpy()
    # <<< SOLUTION


@torch.no_grad()
def zero_shot(model, vocab, imgs, labels) -> float:
    prompts = [f"a {c} {s}" for c in COLORS for s in SHAPES]
    t = model.encode_text(vocab.encode(prompts).to(DEVICE))
    pred = (model.encode_image(to_tensor(imgs).to(DEVICE)) @ t.t()).argmax(1).cpu().numpy()
    truth = [prompts.index(f"a {l['color']} {l['shape']}") for l in labels]
    return float((pred == np.array(truth)).mean())


def heuristic_keep(caps: list[str]) -> np.ndarray:
    words = set(COLORS) | set(SHAPES)
    return np.array([sum(w in words for w in re.findall(r"[a-z]+", c.lower())) >= 2 for c in caps])


def keep_top(scores: np.ndarray, frac: float) -> np.ndarray:
    # >>> SOLUTION: keep the top `frac` by score (ties broken arbitrarily)
    k = int(round(frac * len(scores)))
    keep = np.zeros(len(scores), bool)
    keep[np.argsort(-scores)[:k]] = True
    return keep
    # <<< SOLUTION


def evaluate(keep: np.ndarray, labels) -> dict:
    ok = np.array([l["ok"] for l in labels])
    tp = (keep & ok).sum()
    return {"kept": int(keep.sum()), "precision": tp / max(1, keep.sum()), "recall of good": tp / ok.sum()}


def report() -> str:
    model, vocab = train()
    imgs, caps, labels = pairs(3000, seed=7, wrong=0.25, junk=0.15)
    s = clip_scores(model, vocab, imgs, caps)
    rows = [{"filter": "none", **evaluate(np.ones(len(caps), bool), labels)},
            {"filter": "caption heuristic", **evaluate(heuristic_keep(caps), labels)}]
    for q in (0.3, 0.5, 0.6):
        rows.append({"filter": f"CLIP top {int(q * 100)}%", **evaluate(keep_top(s, q), labels)})
    zs = zero_shot(model, vocab, *[pairs(500, seed=9)[i] for i in (0, 2)])
    return (f"device={DEVICE}; zero-shot 16-way accuracy on held-out images: {zs:.3f}\n"
            "noisy pool: 3000 pairs, 25% wrong captions, 15% junk alt-text (60% good)\n" +
            table(rows, ["filter", "kept", "precision", "recall of good"]))


if __name__ == "__main__":
    print(report())
