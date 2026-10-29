# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 13: CUDA depth - warp MMA fragments, bank conflicts, swizzling, pipelining.

Roadmap: "Warp-level MMA, async copies, shared-memory swizzling. Read one
Hopper-specific kernel (FlashAttention-3 or DeepGEMM) line by line."

These are the three hardware facts every fast kernel is built around. The CPU
part models each one exactly; the GPU part (mma_check.cu) runs a real
mma.sync.m16n8k16 and checks your fragment layout against the hardware.

  1. Fragment layouts of mma.sync.aligned.m16n8k16 (f16 in, f32 accumulate),
     from the PTX ISA "Matrix Fragments for mma.m16n8k16" tables.
     groupID = lane >> 2, tig (thread-in-group) = lane % 4.
  2. Shared memory has 32 banks of 4 bytes. A warp's access costs as many
     passes as the most-loaded bank has *distinct* words (same word = broadcast).
  3. cp.async / TMA pipelining overlaps tile loads with compute; with >= 2 stages
     the steady state costs max(load, compute) per tile instead of the sum.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch


# ------------------------------------------------------------ 1. MMA fragments

def a_frag(lane: int, i: int) -> tuple[int, int]:
    """(row, col) in the 16x16 A tile held by `lane` in half-element i (0..7)."""
    # TODO: row = groupID + 8*((i>>1)&1); col = tig*2 + (i&1) + 8*(i>>2)
    raise NotImplementedError("TODO(w13_cuda_depth): row = groupID + 8*((i>>1)&1); col = tig*2 + (i&1) + 8*(i>>2)")


def b_frag(lane: int, i: int) -> tuple[int, int]:
    """(k, n) in the 16x8 B tile held by `lane` in half-element i (0..3)."""
    # TODO: k = tig*2 + (i&1) + 8*(i>>1); n = groupID
    raise NotImplementedError("TODO(w13_cuda_depth): k = tig*2 + (i&1) + 8*(i>>1); n = groupID")


def c_frag(lane: int, i: int) -> tuple[int, int]:
    """(row, col) in the 16x8 C/D tile held by `lane` in float i (0..3)."""
    # TODO: row = groupID + 8*(i>>1); col = tig*2 + (i&1)
    raise NotImplementedError("TODO(w13_cuda_depth): row = groupID + 8*(i>>1); col = tig*2 + (i&1)")


def simulate_mma(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Scatter A, B, C into 32 lanes' registers, then let 'hardware' compute.

    The hardware needs every lane's registers; we reassemble from registers
    (proving the layout covers each element exactly once), multiply, and
    redistribute D. A wrong layout produces a wrong D."""
    regs_a = [[A[a_frag(l, i)] for i in range(8)] for l in range(32)]
    regs_b = [[B[b_frag(l, i)] for i in range(4)] for l in range(32)]
    regs_c = [[C[c_frag(l, i)] for i in range(4)] for l in range(32)]
    A2, B2, C2 = torch.zeros_like(A), torch.zeros_like(B), torch.zeros_like(C)
    for l in range(32):
        for i in range(8):
            A2[a_frag(l, i)] = regs_a[l][i]
        for i in range(4):
            B2[b_frag(l, i)] = regs_b[l][i]
            C2[c_frag(l, i)] = regs_c[l][i]
    D = A2.float() @ B2.float() + C2
    out = torch.zeros_like(C)
    for l in range(32):
        for i in range(4):
            out[c_frag(l, i)] = D[c_frag(l, i)]
    return out


# ------------------------------------------------------------ 2. banks

def bank_conflict_degree(word_addrs: list[int]) -> int:
    """Passes needed for one warp access to 32-bit words at these addresses."""
    # TODO: bank = addr % 32; for each bank count distinct addresses; return the max (at least 1)
    raise NotImplementedError("TODO(w13_cuda_depth): bank = addr % 32; for each bank count distinct addresses; return the max (at least 1)")


def column_read(row_stride: int, col: int = 0, swizzle: bool = False) -> list[int]:
    """32 lanes read column `col` of a 32-row float tile (lane = row)."""
    return [r * row_stride + (swizzled_col(r, col) if swizzle else col) for r in range(32)]


def swizzled_col(row: int, col: int) -> int:
    """XOR swizzle: store element (row, col) at physical column col ^ row (mod 32).

    Rows still hold a permutation of their 32 columns (so row reads stay
    conflict-free), but a column now lands in 32 different banks - without the
    wasted memory of padding."""
    # TODO: col ^ (row % 32)
    raise NotImplementedError("TODO(w13_cuda_depth): col ^ (row % 32)")


# ------------------------------------------------------------ 3. pipelining

def pipeline_time(n_tiles: int, load: float, compute: float, stages: int) -> float:
    """Time to process n tiles. stages=1: load then compute, serially.
    stages>=2: prologue load, then each step overlaps the next load with this compute."""
    # TODO: stages==1 -> n*(load+compute); else load + (n-1)*max(load, compute) + compute
    raise NotImplementedError("TODO(w13_cuda_depth): stages==1 -> n*(load+compute); else load + (n-1)*max(load, compute) + compute")


# ------------------------------------------------------------ GPU check

def gpu_mma_check() -> float:
    """Compile mma_check.cu (inline PTX mma.sync) and compare with A @ B + C.
    The kernel uses *your* fragment functions' layout, mirrored in C++."""
    from torch.utils.cpp_extension import load

    here = pathlib.Path(__file__).resolve().parent
    ext = load(name="w13_mma", sources=[str(here / "mma_check.cu")], extra_cuda_cflags=["-arch=sm_80"], verbose=False)
    A = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    B = torch.randn(16, 8, device="cuda", dtype=torch.float16)
    C = torch.randn(16, 8, device="cuda", dtype=torch.float32)
    D = ext.mma_16x8x16(A, B, C)
    return (D - (A.float() @ B.float() + C)).abs().max().item()


if __name__ == "__main__":
    A, B, C = torch.randn(16, 16), torch.randn(16, 8), torch.randn(16, 8)
    print("simulated mma max |diff|:", (simulate_mma(A, B, C) - (A @ B + C)).abs().max().item())
    for name, addrs in [("naive 32x32", column_read(32)), ("padded 32x33", column_read(33)),
                        ("xor-swizzled", column_read(32, swizzle=True))]:
        print(f"column read, {name:13s}: {bank_conflict_degree(addrs)}-way")
    print("16 tiles, load 1.0, compute 0.8: 1 stage", pipeline_time(16, 1, 0.8, 1), "| 2 stages", pipeline_time(16, 1, 0.8, 2))
    if torch.cuda.is_available():
        print("hardware mma.sync max |diff|:", gpu_mma_check())
