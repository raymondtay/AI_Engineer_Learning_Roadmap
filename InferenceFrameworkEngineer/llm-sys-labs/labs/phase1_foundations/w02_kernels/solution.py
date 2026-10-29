"""Week 2: CUDA basics (vector add, reduction, tiled SGEMM).

On a GPU this compiles `<prefix>_kernels.cu` with nvcc. On CPU it compiles the
twin `<prefix>_kernels.cpp`, written in the same block/thread/tile structure.
`<prefix>` is "solution" or "exercise", matching this file's name, so the
exercise version builds your edited kernels.

Gate (roadmap): tiled SGEMM at 4096^2 reaching >= 70% of cuBLAS on the GPU.
CPU equivalent reported here: % of torch.matmul (MKL/OpenBLAS) at 1024^2.

Run:  python solution.py --bench
Profile on GPU:  ncu --set full -k regex:sgemm python solution.py --bench
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
from torch.utils.cpp_extension import load

from common.bench import bench, table
from common.device import device

HERE = pathlib.Path(__file__).resolve().parent
PREFIX = pathlib.Path(__file__).stem  # "solution" or "exercise"
_EXT = None


def build():
    """Compile and cache the extension for the active device."""
    global _EXT
    if _EXT is not None:
        return _EXT
    build_dir = HERE / "build" / f"{PREFIX}_{device().type}"
    build_dir.mkdir(parents=True, exist_ok=True)
    if device().type == "cuda":
        _EXT = load(
            name=f"w02_{PREFIX}_cuda",
            sources=[str(HERE / f"{PREFIX}_kernels.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            build_directory=str(build_dir),
            verbose=False,
        )
    else:
        _EXT = load(
            name=f"w02_{PREFIX}_cpu",
            sources=[str(HERE / f"{PREFIX}_kernels.cpp")],
            extra_cflags=["-O3", "-fopenmp", "-march=native"],
            extra_ldflags=["-fopenmp"],
            build_directory=str(build_dir),
            verbose=False,
        )
    return _EXT


def vector_add(a, b):
    return build().vector_add(a.contiguous(), b.contiguous())


def reduce_sum(x):
    return build().reduce_sum(x.contiguous())


def sgemm(a, b):
    return build().sgemm(a.contiguous(), b.contiguous())


def bench_sgemm(n: int) -> dict:
    dev = device()
    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    t_ours = bench(lambda: sgemm(a, b), warmup=1, iters=3)
    t_ref = bench(lambda: a @ b, warmup=1, iters=3)
    flops = 2 * n**3
    return {
        "n": n,
        "ours GFLOP/s": flops / t_ours / 1e9,
        "library GFLOP/s": flops / t_ref / 1e9,
        "% of library": 100 * t_ref / t_ours,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    build()
    sizes = [512, 1024] if device().type == "cpu" else [1024, 2048, 4096]
    rows = [bench_sgemm(n) for n in sizes]
    print(f"device={device()}")
    print(table(rows, ["n", "ours GFLOP/s", "library GFLOP/s", "% of library"]))
