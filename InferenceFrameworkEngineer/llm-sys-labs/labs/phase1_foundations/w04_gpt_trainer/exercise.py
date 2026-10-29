# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 4: a GPT you own end to end.

Roadmap: "Train GPT-2 124M from your own code on one GPU. Add BF16, activation
checkpointing, torch.compile; measure each change."

CPU default: a ~100K-parameter model on a built-in byte corpus (trains in
seconds). GPU: `--preset gpt2` gives the 124M shape (12 layers, 12 heads,
768 wide, 1024 context); feed it real data with --data path/to/text.txt.

Run:
  python solution.py                       # CPU-sized ablation table
  python solution.py --preset gpt2 --steps 200 --data input.txt   # on a GPU
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time
from dataclasses import dataclass, replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.bench import table
from common.device import autocast_dtype, device
from common.flops import PEAK_TFLOPS, cpu_peak_tflops_estimate, mfu, train_flops_per_token
from common.tinygpt import CORPUS, batches, encode


@dataclass
class Config:
    vocab_size: int = 256
    seq_len: int = 64
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 64
    batch: int = 16
    steps: int = 150
    lr: float = 3e-3
    warmup: int = 10
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bf16: bool = False
    checkpoint: bool = False
    compile: bool = False
    seed: int = 0


PRESETS = {
    "tiny": Config(),
    "gpt2": Config(seq_len=1024, n_layer=12, n_head=12, d_model=768, batch=8, lr=6e-4, warmup=100, steps=1000, vocab_size=50304),
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, use_sdpa: bool = True):
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.qkv(x).view(B, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
        if use_sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # TODO: scores = q@k^T / sqrt(d); mask the upper triangle to -inf; softmax in fp32; @ v
            raise NotImplementedError("TODO(w04_gpt_trainer): scores = q@k^T / sqrt(d); mask the upper triangle to -inf; softmax in fp32; @ v")
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.d_model), CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False), nn.GELU(),
                                 nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight
        for n, p in self.named_parameters():
            if p.dim() >= 2:
                # GPT-2 init: residual projections scaled by 1/sqrt(2 * n_layer)
                std = 0.02 / math.sqrt(2 * cfg.n_layer) if n.endswith(("proj.weight", "mlp.2.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.cfg.checkpoint and self.training) else blk(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss

    def non_embedding_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.wte.weight.numel() - self.wpe.weight.numel()


def lr_at(step: int, cfg: Config) -> float:
    """Linear warmup to cfg.lr, then cosine decay to 10% of cfg.lr at cfg.steps."""
    # TODO: warmup: lr*(step+1)/warmup; then 0.1*lr + 0.9*lr*0.5*(1+cos(pi*progress))
    raise NotImplementedError("TODO(w04_gpt_trainer): warmup: lr*(step+1)/warmup; then 0.1*lr + 0.9*lr*0.5*(1+cos(pi*progress))")


def make_optimizer(model: nn.Module, cfg: Config):
    """AdamW with weight decay only on matrices (not norms, not biases)."""
    # TODO: two param groups: dim>=2 gets weight_decay, the rest 0.0; fused AdamW on CUDA
    raise NotImplementedError("TODO(w04_gpt_trainer): two param groups: dim>=2 gets weight_decay, the rest 0.0; fused AdamW on CUDA")


def train_step(model, opt, x, y, cfg: Config, step: int) -> float:
    """One optimisation step. Returns the loss as a float."""
    # TODO: set lr; forward under autocast(bf16) if cfg.bf16; backward; clip grad norm; step; zero grads
    raise NotImplementedError("TODO(w04_gpt_trainer): set lr; forward under autocast(bf16) if cfg.bf16; backward; clip grad norm; step; zero grads")


def peak_tflops() -> float:
    if device().type == "cuda":
        name = torch.cuda.get_device_name()
        for (gpu, prec), v in PEAK_TFLOPS.items():
            if prec == "bf16" and gpu.split("-")[0] in name.replace(" ", ""):
                return v
        return float(os.environ.get("PEAK_TFLOPS", "100"))
    return cpu_peak_tflops_estimate(os.cpu_count() or 1)


def train(cfg: Config, data: torch.Tensor | None = None, log_every: int = 0) -> dict:
    torch.manual_seed(cfg.seed)
    dev = device()
    data = encode(CORPUS) if data is None else data
    model = GPT(cfg).to(dev)
    opt = make_optimizer(model, cfg)
    step_model = torch.compile(model) if cfg.compile else model
    gen = torch.Generator().manual_seed(cfg.seed)
    it = batches(data, cfg.batch, cfg.seq_len, generator=gen)
    losses, t0, timed_tokens = [], None, 0
    for step in range(cfg.steps):
        x, y = (t.to(dev) for t in next(it))
        if step == 5:  # exclude warm-up / compilation from throughput
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
        losses.append(train_step(step_model, opt, x, y, cfg, step))
        if step >= 5:
            timed_tokens += x.numel()
        if log_every and step % log_every == 0:
            print(f"step {step:5d}  loss {losses[-1]:.4f}  lr {lr_at(step, cfg):.2e}")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tps = timed_tokens / dt
    fpt = train_flops_per_token(model.non_embedding_params(), cfg.n_layer, cfg.d_model, cfg.seq_len)
    return {"final_loss": sum(losses[-10:]) / 10, "tokens/s": tps, "MFU %": 100 * mfu(tps, fpt, peak_tflops()), "losses": losses}


def ablation(base: Config) -> list[dict]:
    """Measure each optimisation on its own, as the roadmap gate asks."""
    variants = {
        "baseline fp32": base,
        "+ bf16 autocast": replace(base, bf16=True),
        "+ activation checkpointing": replace(base, checkpoint=True),
        "+ torch.compile": replace(base, compile=True),
    }
    rows = []
    for name, c in variants.items():
        r = train(c)
        rows.append({"variant": name, "final loss": r["final_loss"], "tokens/s": r["tokens/s"], "MFU %": r["MFU %"]})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=PRESETS)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--data", help="UTF-8 text file; bytes are the tokens")
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    if args.steps:
        cfg = replace(cfg, steps=args.steps)
    data = encode(pathlib.Path(args.data).read_text()) if args.data else None
    if data is not None and cfg.vocab_size != 256:
        cfg = replace(cfg, vocab_size=256)  # byte-level data
    print(f"device={device()}  peak used for MFU={peak_tflops():.2f} TFLOP/s")
    rows = ablation(cfg) if data is None else [train(cfg, data, log_every=50)]
    print(table(rows, ["variant", "final loss", "tokens/s", "MFU %"]) if data is None else rows[0]["final_loss"])
