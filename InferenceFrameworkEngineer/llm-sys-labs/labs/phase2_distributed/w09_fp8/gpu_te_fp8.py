"""Week 9 (GPU only): the same FP8-vs-BF16 comparison with NVIDIA Transformer Engine.

STATUS: not executed by the author (no GPU in the build environment). Uses the
documented TE PyTorch API: te.Linear, te.fp8_autocast, and the DelayedScaling
recipe with Format.HYBRID (E4M3 forward, E5M2 backward). Newer TE releases also
offer block-scaled recipes for Blackwell; check `transformer_engine.common.recipe`
in your installed version.

Requires: Hopper/Ada/Blackwell GPU, `pip install transformer_engine[pytorch]`.
Run:      python gpu_te_fp8.py --steps 2000
Gate:     |mean loss FP8 - mean loss BF16| / BF16 < 0.25% over the last 10% of steps.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = te.Linear(d, 3 * d, bias=False), te.Linear(d, d, bias=False)
        self.up, self.down = te.Linear(d, 4 * d, bias=False), te.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class Model(nn.Module):
    def __init__(self, vocab=50304, d=1024, h=16, layers=12):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(layers))
        self.head = nn.Linear(d, vocab, bias=False)  # kept in BF16, as DeepSeek-V3 does

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def run(fp8: bool, steps: int, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    model = Model().cuda().to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
    g = torch.Generator(device="cuda").manual_seed(seed)
    losses = []
    for _ in range(steps):
        # Replace with real tokens for a meaningful curve (random tokens only test stability).
        idx = torch.randint(0, 50304, (8, 1025), device="cuda", generator=g)
        with te.fp8_autocast(enabled=fp8, fp8_recipe=recipe):
            logits = model(idx[:, :-1])
        loss = F.cross_entropy(logits.float().flatten(0, 1), idx[:, 1:].flatten())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    base, f8 = run(False, a.steps), run(True, a.steps)
    tail = max(10, a.steps // 10)
    b, f = sum(base[-tail:]) / tail, sum(f8[-tail:]) / tail
    print(f"bf16 {b:.4f}  fp8 {f:.4f}  rel gap {abs(f - b) / b:.4%}  (gate: < 0.25%)")
