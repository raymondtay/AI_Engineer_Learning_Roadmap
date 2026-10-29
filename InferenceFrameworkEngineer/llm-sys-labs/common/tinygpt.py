"""Reference GPT used by Phases 2-6.

This is the finished version of what you build in Week 4. Later phases import
it so that a bug in your Week 4 exercise does not block Week 14. Byte-level
vocabulary (256) so no tokenizer download is needed.

Hooks later labs rely on:
  * `embeds=`   feed pre-computed embeddings (Week 22 VLM)
  * `pos=`      explicit position ids (Week 23 sequence packing)
  * `attn_mask=` boolean mask, True = may attend (Week 23 packing)
  * Block.attn exposes q/k/v projection so the inference engine can run its
    own paged attention against the same weights (Week 14).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256
    seq_len: int = 128
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 64
    mlp_ratio: int = 4

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary embedding. x: (B, H, T, D), pos: (B, T) or (T,) integer positions."""
    d = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    if pos.dim() == 1:
        pos = pos[None]
    ang = pos[:, None, :, None].float() * inv  # (B,1,T,D/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def split_qkv(self, x, pos):
        B, T, _ = x.shape
        H, D = self.cfg.n_head, self.cfg.head_dim
        q, k, v = self.qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        return rope(q, pos), rope(k, pos), v  # each (B,H,T,D)

    def forward(self, x, pos, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.split_qkv(x, pos)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[:, None])
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.mlp_ratio * cfg.d_model, bias=False)
        self.down = nn.Linear(cfg.mlp_ratio * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.ln2, self.mlp = RMSNorm(cfg.d_model), MLP(cfg)

    def forward(self, x, pos, attn_mask=None):
        x = x + self.attn(self.ln1(x), pos, attn_mask)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx=None, targets=None, *, embeds=None, pos=None, attn_mask=None, loss_mask=None):
        x = self.wte(idx) if embeds is None else embeds
        B, T, _ = x.shape
        if pos is None:
            pos = torch.arange(T, device=x.device)
        for blk in self.blocks:
            x = blk(x, pos, attn_mask)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none")
        if loss_mask is not None:
            m = loss_mask.flatten().float()
            return logits, (loss * m).sum() / m.sum().clamp(min=1)
        return logits, loss.mean()

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.wte.weight.numel() if non_embedding else n

    @torch.no_grad()
    def forward_with_past(self, tokens: torch.Tensor, offset: int, past=None):
        """Run tokens at positions [offset, offset+T) given per-layer past (k, v)
        of shape (1, H, offset, D). Returns (logits of last token, per-layer (k, v)
        of *these* tokens). Used by the KV-cache labs (Weeks 15-17)."""
        x = self.wte(tokens.view(1, -1))
        T = x.shape[1]
        pos = torch.arange(offset, offset + T, device=x.device)
        new_kv = []
        for li, blk in enumerate(self.blocks):
            q, k, v = blk.attn.split_qkv(blk.ln1(x), pos)
            new_kv.append((k, v))
            K = k if past is None else torch.cat([past[li][0], k], 2)
            V = v if past is None else torch.cat([past[li][1], v], 2)
            mask = torch.arange(offset + T, device=x.device)[None, :] <= pos[:, None]
            y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
            x = x + blk.attn.proj(y.transpose(1, 2).reshape(1, T, -1))
            x = x + blk.mlp(blk.ln2(x))
        return self.head(self.ln_f(x[:, -1]))[0], new_kv

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, generator=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.seq_len:])
            logits = logits[:, -1].float() / max(temperature, 1e-5)
            nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)
            idx = torch.cat([idx, nxt], 1)
        return idx


# ---------------------------------------------------------------- data helpers

CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
) * 64


def encode(s: str) -> torch.Tensor:
    return torch.tensor(list(s.encode("utf-8")), dtype=torch.long)


def decode(t) -> str:
    return bytes(int(i) % 256 for i in t).decode("utf-8", errors="replace")


def batches(data: torch.Tensor, batch: int, seq_len: int, generator=None):
    """Infinite iterator of (x, y) next-token batches from a 1-D byte tensor."""
    while True:
        ix = torch.randint(len(data) - seq_len - 1, (batch,), generator=generator)
        x = torch.stack([data[i : i + seq_len] for i in ix])
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])
        yield x, y
