# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 10: Triton - fused softmax, RMSNorm forward/backward, autotuned matmul.

Roadmap: "Fused softmax, LayerNorm/RMSNorm forward and backward, matmul with
autotune. Benchmark against PyTorch eager and torch.compile."

CPU: kernels run under TRITON_INTERPRET=1 (set automatically by common/), which
executes the same kernel source with NumPy. Correctness is real; speed is not -
benchmark on a GPU.

Each kernel is one Triton "program" per row (softmax, RMSNorm) or per output
tile (matmul). The point of fusing: softmax in eager PyTorch reads and writes
HBM ~5 times; the fused kernel reads once and writes once.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import common  # noqa: F401  (sets TRITON_INTERPRET on CPU before triton import)
import torch
import triton
import triton.language as tl

from common.device import is_cuda

# ------------------------------------------------------------------ softmax


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # TODO: load the row (masked lanes = -inf), subtract max, exp, divide by sum, store
    raise NotImplementedError("TODO(w10_triton_basics): load the row (masked lanes = -inf), subtract max, exp, divide by sum, store")


def softmax(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    R, C = x.shape
    y = torch.empty_like(x)
    softmax_kernel[(R,)](x, y, C, x.stride(0), BLOCK=triton.next_power_of_2(C))
    return y


# ------------------------------------------------------------------ RMSNorm


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, n_cols, stride, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # TODO: rstd = 1/sqrt(mean(x^2)+eps) in fp32; store rstd[row]; y = x*rstd*w
    raise NotImplementedError("TODO(w10_triton_basics): rstd = 1/sqrt(mean(x^2)+eps) in fp32; store rstd[row]; y = x*rstd*w")


@triton.jit
def rmsnorm_bwd_kernel(g_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_part_ptr, n_cols, stride, BLOCK: tl.constexpr):
    """dx per row; per-row contribution to dw written to dw_part[row] (summed on host)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    # TODO: gw = g*w; dx = rstd*gw - x*rstd^3*mean(gw*x); dw_part = g*x*rstd
    raise NotImplementedError("TODO(w10_triton_basics): gw = g*w; dx = rstd*gw - x*rstd^3*mean(gw*x); dw_part = g*x*rstd")


class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps=1e-6):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        R, C = x2.shape
        y = torch.empty_like(x2)
        rstd = torch.empty(R, device=x.device, dtype=torch.float32)
        BLOCK = triton.next_power_of_2(C)
        rmsnorm_fwd_kernel[(R,)](x2, w, y, rstd, C, x2.stride(0), eps, BLOCK=BLOCK)
        ctx.save_for_backward(x2, w, rstd)
        ctx.shp, ctx.BLOCK = shp, BLOCK
        return y.reshape(shp)

    @staticmethod
    def backward(ctx, g):
        x2, w, rstd = ctx.saved_tensors
        g2 = g.reshape(x2.shape).contiguous()
        dx = torch.empty_like(x2)
        dw_part = torch.empty_like(x2, dtype=torch.float32)
        R, C = x2.shape
        rmsnorm_bwd_kernel[(R,)](g2, x2, w, rstd, dx, dw_part, C, x2.stride(0), BLOCK=ctx.BLOCK)
        return dx.reshape(ctx.shp), dw_part.sum(0).to(w.dtype), None


def rmsnorm(x, w, eps=1e-6):
    return TritonRMSNorm.apply(x, w, eps)


# ------------------------------------------------------------------ matmul

# On GPU, autotune picks the best tile shape per (M, N, K). Under the
# interpreter one small config keeps tests fast.
_CONFIGS = (
    [triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
     triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=4, num_stages=4),
     triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=4, num_stages=4),
     triton.Config({"BM": 64, "BN": 64, "BK": 64}, num_warps=4, num_stages=3)]
    if is_cuda()
    else [triton.Config({"BM": 32, "BN": 32, "BK": 32})]
)


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # TODO: loop over K in BK steps; masked loads of A[rm, k] and B[k, rn]; acc += tl.dot(a, b)
    raise NotImplementedError("TODO(w10_triton_basics): loop over K in BK steps; masked loads of A[rm, k] and B[k, rn]; acc += tl.dot(a, b)")
    tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc.to(c_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


def benchmark() -> str:
    """Eager vs torch.compile vs Triton. Meaningful only on a GPU."""
    from common.bench import bench, table
    from common.device import device

    dev = device()
    n = 4096 if is_cuda() else 256
    x = torch.randn(n, n, device=dev)
    w = torch.ones(n, device=dev)
    rows = []
    for name, eager, ours in [
        ("softmax", lambda: torch.softmax(x, -1), lambda: softmax(x)),
        ("rmsnorm", lambda: torch.nn.functional.rms_norm(x, (n,), w), lambda: rmsnorm(x, w)),
        ("matmul", lambda: x @ x, lambda: matmul(x, x)),
    ]:
        compiled = torch.compile(eager)
        rows.append({"op": name, "eager us": bench(eager) * 1e6, "compile us": bench(compiled) * 1e6,
                     "triton us": bench(ours, warmup=1, iters=3) * 1e6})
    note = "" if is_cuda() else "\n(CPU: Triton runs in the interpreter, so its times are not meaningful.)"
    return table(rows, ["op", "eager us", "compile us", "triton us"]) + note


if __name__ == "__main__":
    print(benchmark())
