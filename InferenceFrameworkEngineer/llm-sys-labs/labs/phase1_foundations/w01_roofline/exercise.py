# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 1: GPU architecture and the roofline model.

Roadmap: "Compute arithmetic intensity for GEMM, softmax and LayerNorm by hand."

The roofline says a kernel's best-case time is
    t >= max(FLOPs / peak_FLOP/s,  bytes_moved / peak_bandwidth)
Arithmetic intensity (AI) = FLOPs / byte. Below the ridge point
(peak_FLOP/s / bandwidth) a kernel is memory-bound; above it, compute-bound.

Conventions used throughout (state yours when you report numbers):
  * a fused multiply-add counts as 2 FLOPs
  * bytes = each input read once + each output written once (ideal caching)
  * softmax: 5 FLOPs/element (max, subtract, exp, sum, divide)
  * LayerNorm: 8 FLOPs/element (mean, var: sub+square+sum, normalise, scale, shift)

Run:  python labs/phase1_foundations/w01_roofline/solution.py
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from common.bench import bench, table
from common.device import device


def ai_gemm(M: int, N: int, K: int, bytes_per_el: int) -> float:
    """C[M,N] = A[M,K] @ B[K,N]."""
    # TODO: FLOPs = 2*M*N*K; bytes = (M*K + K*N + M*N) * bytes_per_el
    raise NotImplementedError("TODO(w01_roofline): FLOPs = 2*M*N*K; bytes = (M*K + K*N + M*N) * bytes_per_el")


def ai_softmax(rows: int, cols: int, bytes_per_el: int) -> float:
    # TODO: 5 FLOPs per element; read input once, write output once
    raise NotImplementedError("TODO(w01_roofline): 5 FLOPs per element; read input once, write output once")


def ai_layernorm(rows: int, cols: int, bytes_per_el: int) -> float:
    """Includes reading gamma and beta (cols elements each)."""
    # TODO: 8 FLOPs per element; bytes = in + out + gamma + beta
    raise NotImplementedError("TODO(w01_roofline): 8 FLOPs per element; bytes = in + out + gamma + beta")


def ridge_point(peak_flops: float, bandwidth: float) -> float:
    """FLOPs/byte where the roofline turns from memory- to compute-bound."""
    # TODO: peak / bandwidth
    raise NotImplementedError("TODO(w01_roofline): peak / bandwidth")


def roofline_time(flops: float, nbytes: float, peak_flops: float, bandwidth: float) -> tuple[float, str]:
    """Return (lower-bound seconds, 'compute' | 'memory')."""
    # TODO: the slower of the two limits wins
    raise NotImplementedError("TODO(w01_roofline): the slower of the two limits wins")


# ------------------------------------------------------------- measurement

def measure_bandwidth(n_bytes: int = 256 * 2**20) -> float:
    """Achieved copy bandwidth (bytes/s): read n + write n."""
    dev = device()
    x = torch.empty(n_bytes // 4, dtype=torch.float32, device=dev)
    y = torch.empty_like(x)
    t = bench(lambda: y.copy_(x), warmup=2, iters=5)
    return 2 * n_bytes / t


def measure_matmul_flops(n: int = 1024, dtype=torch.float32) -> float:
    dev = device()
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    t = bench(lambda: a @ b, warmup=2, iters=5)
    return 2 * n**3 / t


def report(n: int = 1024) -> str:
    """Measure this machine and place GEMM, softmax and LayerNorm on its roofline."""
    bw = measure_bandwidth()
    peak = measure_matmul_flops(n)
    rows = []
    cases = {
        "gemm": (ai_gemm(n, n, n, 4), 2 * n**3),
        "softmax": (ai_softmax(n, n, 4), 5 * n * n),
        "layernorm": (ai_layernorm(n, n, 4), 8 * n * n),
    }
    for name, (ai, flops) in cases.items():
        t, bound = roofline_time(flops, flops / ai, peak, bw)
        rows.append({"op": name, "AI (FLOP/B)": ai, "bound": bound, "roofline us": t * 1e6})
    head = f"device={device()}  measured bw={bw/1e9:.1f} GB/s  matmul={peak/1e12:.3f} TFLOP/s  ridge={peak/bw:.1f} FLOP/B\n"
    return head + table(rows, ["op", "AI (FLOP/B)", "bound", "roofline us"])


if __name__ == "__main__":
    print(report())
