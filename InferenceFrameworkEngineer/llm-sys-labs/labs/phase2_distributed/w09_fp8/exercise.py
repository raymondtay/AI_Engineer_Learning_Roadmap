# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 9: FP8 training.

Roadmap: "Transformer Engine FP8 with delayed scaling, then block-wise
scaling. Run FP8 vs BF16 on the same model and compare loss curves."
Gate: FP8 loss within 0.25% relative of the higher-precision baseline
(DeepSeek-V3 reports staying below 0.25%).

CPU path: real FP8 rounding via torch.float8_e4m3fn / float8_e5m2 casts, with
the matmul done in fp32 after dequantisation. Values match what an FP8 GEMM
with fp32 accumulation computes; only the speed is emulated. On Hopper, run
gpu_te_fp8.py next to this file for the Transformer Engine version.

Parts:
  1. quantize / dequantize with per-tensor or block-wise scales
  2. DelayedScaler - per-tensor scale from an amax history (TE's default recipe)
  3. FP8Linear     - forward E4M3, backward grads E5M2 ("hybrid" format)
  4. compare_training - FP8 vs fp32 loss on the shared tiny GPT
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.tinygpt import CORPUS, GPT, GPTConfig, batches, encode

FMT = {
    "e4m3": (torch.float8_e4m3fn, 448.0),
    "e5m2": (torch.float8_e5m2, 57344.0),
}


def quantize(x: torch.Tensor, fmt: str = "e4m3", block: tuple[int, int] | None = None):
    """Return (q, scale) with x ~= q.float() * scale.

    block=None: one scale for the whole tensor.
    block=(br, bc): one scale per br x bc tile of a 2-D tensor (pad-free: dims must divide).
    DeepSeek-V3 uses (1, 128) tiles for activations and (128, 128) for weights."""
    dtype, fmax = FMT[fmt]
    # TODO: scale = amax / fmax (clamped away from 0); q = (x / scale).to(fp8 dtype); for blocks reshape to (R/br, br, C/bc, bc) and take amax per tile
    raise NotImplementedError("TODO(w09_fp8): scale = amax / fmax (clamped away from 0); q = (x / scale).to(fp8 dtype); for blocks reshape to (R/br, br, C/bc, bc) and take amax per tile")


def dequantize(q: torch.Tensor, scale: torch.Tensor, block: tuple[int, int] | None = None) -> torch.Tensor:
    if block is None:
        return q.float() * scale
    br, bc = block
    R, C = q.shape
    return (q.float().reshape(R // br, br, C // bc, bc) * scale).reshape(R, C)


def rel_error(x: torch.Tensor, block=None, fmt="e4m3") -> float:
    """Norm-relative error. Beware: dominated by the largest values, so it
    hides what quantisation does to the bulk of small values."""
    q, s = quantize(x, fmt, block)
    return ((dequantize(q, s, block) - x).norm() / x.norm()).item()


def bulk_error(x: torch.Tensor, block=None, fmt="e4m3", outlier_q: float = 0.999) -> tuple[float, float]:
    """(median relative error, fraction flushed to zero) over the non-outlier bulk.

    This is the metric that shows why one outlier ruins a per-tensor scale:
    the scale stretches to fit the outlier and ordinary values underflow."""
    # TODO: bulk = nonzero |x| below the outlier_q quantile; median of |dq - x|/|x| over bulk; mean of (dq == 0) over bulk
    raise NotImplementedError("TODO(w09_fp8): bulk = nonzero |x| below the outlier_q quantile; median of |dq - x|/|x| over bulk; mean of (dq == 0) over bulk")


class DelayedScaler:
    """Scale from the max amax over the last `history` steps (TE DelayedScaling).

    Uses the *previous* steps' statistics so the current tensor needs no extra
    reduction pass before casting - the reason it is the default on Hopper."""

    def __init__(self, history: int = 16, fmt: str = "e4m3", margin: float = 0.0):
        self.hist: list[float] = []
        self.n, self.fmt, self.margin = history, fmt, margin

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: use max(history) (or current amax on first call) / fmax * 2**margin; then append current amax, trim to n
        raise NotImplementedError("TODO(w09_fp8): use max(history) (or current amax on first call) / fmax * 2**margin; then append current amax, trim to n")


class _FP8MatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, w, xb, wb):
        # TODO: quantize x with tiles xb and w with tiles wb (E4M3), matmul the dequantized values in fp32, save quantized copies
        raise NotImplementedError("TODO(w09_fp8): quantize x with tiles xb and w with tiles wb (E4M3), matmul the dequantized values in fp32, save quantized copies")

    @staticmethod
    def backward(ctx, g):
        # TODO: quantize g per-tensor in E5M2 (wide range for gradients); dx = g @ w; dw = g^T @ x
        raise NotImplementedError("TODO(w09_fp8): quantize g per-tensor in E5M2 (wide range for gradients); dx = g @ w; dw = g^T @ x")


class FP8Linear(nn.Linear):
    """Drop-in nn.Linear that runs its GEMMs through FP8 rounding."""

    def __init__(self, *a, x_block=(1, 32), w_block=(32, 32), **kw):
        super().__init__(*a, **kw)
        self.x_block, self.w_block = x_block, w_block

    def forward(self, x):
        shp = x.shape
        y = _FP8MatMul.apply(x.reshape(-1, shp[-1]), self.weight, self.x_block, self.w_block)
        y = y.reshape(*shp[:-1], self.out_features)
        return y if self.bias is None else y + self.bias


def to_fp8(model: nn.Module, skip=("head",), **kw) -> nn.Module:
    """Swap Linear layers for FP8Linear, keeping the same Parameter objects.

    Like DeepSeek-V3, the output head (and embeddings) stay in high precision."""
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and not isinstance(child, FP8Linear) and cname not in skip:
                new = FP8Linear(child.in_features, child.out_features, bias=child.bias is not None, **kw)
                new.weight = child.weight
                if child.bias is not None:
                    new.bias = child.bias
                setattr(mod, cname, new)
    return model


def train_losses(fp8: bool, steps: int = 200, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    model = GPT(GPTConfig(seq_len=64))
    if fp8:
        to_fp8(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    it = batches(encode(CORPUS), 16, 64, generator=torch.Generator().manual_seed(seed))
    out = []
    for _ in range(steps):
        x, y = next(it)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        out.append(loss.item())
    return out


def compare_training(steps: int = 150) -> dict:
    """Relative gap of the mean loss over the whole run (area under the curve).

    Measured on this CPU emulation (tiny GPT, 150 steps): FP8 trails fp32 by a
    few percent while the loss is falling fast and nearly catches up by the
    end. Exercise: find which choice causes the lag (E5M2 per-tensor grads?
    block size? quantising qkv?) by ablating one at a time."""
    base, fp8 = train_losses(False, steps), train_losses(True, steps)
    b, f = sum(base) / steps, sum(fp8) / steps
    tail = max(10, steps // 10)
    return {"baseline_mean": b, "fp8_mean": f, "rel_gap": abs(f - b) / b,
            "baseline_final": sum(base[-tail:]) / tail, "fp8_final": sum(fp8[-tail:]) / tail}


if __name__ == "__main__":
    torch.manual_seed(0)
    for outlier in (1e3, 1e4, 1e5):
        x = torch.randn(128, 128)
        x[0, 0] = outlier  # one large activation, as LLMs often have
        (mt, zt), (mb, zb) = bulk_error(x), bulk_error(x, (1, 128))
        print(f"outlier {outlier:8.0f}: per-tensor median err {mt:6.2%} zeroed {zt:6.2%} | "
              f"block 1x128 median err {mb:6.2%} zeroed {zb:6.2%}")
    print(compare_training())
