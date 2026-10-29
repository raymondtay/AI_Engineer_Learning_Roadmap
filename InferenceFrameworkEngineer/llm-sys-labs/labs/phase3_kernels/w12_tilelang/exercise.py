# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 12: TileLang - GEMM, FP8 block-scaled GEMM, attention.

Roadmap: "Port your GEMM and attention to TileLang; add an FP8 block-scaled
GEMM and an MLA decode kernel from its examples."

Two programs per kernel, same algorithm:
  *_cpu  scalar tile programs (T.alloc_local, T.serial, T.grid, T.copy, T.exp,
         T.max). They compile with target="c" and run on CPU. Verified here
         against PyTorch on TileLang 0.1.15.
  *_gpu  the hardware version: T.alloc_shared / T.alloc_fragment, T.gemm (tensor
         cores), T.Pipelined (async copy pipelining). These need target="cuda"
         (or "hip"); the C backend cannot infer fragment layouts, so they are
         GPU-tested only.

For the MLA decode kernel and a full FlashAttention with T.gemm, start from
the official examples in the tile-ai/tilelang repo (examples/flash_attention,
examples/deepseek_mla) and compare against your Week 11 Triton kernel.
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

try:
    import tilelang
    import tilelang.language as T
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install tilelang  (see requirements-cpu.txt)") from e

TARGET = "cuda" if torch.cuda.is_available() else "c"


def compile(prog, target: str | None = None):
    return tilelang.compile(prog, target=target or TARGET)


# ---------------------------------------------------------------- GEMM


def gemm_cpu(M: int, N: int, K: int, bm: int = 32, bn: int = 32, bk: int = 32):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float32"), B: T.Tensor((K, N), "float32"), C: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=1) as (bx, by):
            A_t = T.alloc_local((bm, bk), "float32")
            B_t = T.alloc_local((bk, bn), "float32")
            C_t = T.alloc_local((bm, bn), "float32")
            T.clear(C_t)
            for ko in T.serial(T.ceildiv(K, bk)):
                # TODO: T.copy the A tile at (by*bm, ko*bk) and the B tile at (ko*bk, bx*bn); accumulate with a T.grid(bm, bn, bk) loop
                raise NotImplementedError("TODO(w12_tilelang): T.copy the A tile at (by*bm, ko*bk) and the B tile at (ko*bk, bx*bn); accumulate with a T.grid(bm, bn, bk) loop")
            T.copy(C_t, C[by * bm, bx * bn])

    return main


def gemm_gpu(M: int, N: int, K: int, bm: int = 128, bn: int = 128, bk: int = 32, dtype: str = "float16"):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=128) as (bx, by):
            A_s = T.alloc_shared((bm, bk), dtype)
            B_s = T.alloc_shared((bk, bn), dtype)
            C_f = T.alloc_fragment((bm, bn), "float32")
            T.clear(C_f)
            for ko in T.Pipelined(T.ceildiv(K, bk), num_stages=3):
                T.copy(A[by * bm, ko * bk], A_s)
                T.copy(B[ko * bk, bx * bn], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * bm, bx * bn])

    return main


# ---------------------------------------------------------------- FP8 block-scaled GEMM


def blockscaled_gemm_cpu(M: int, N: int, K: int, blk: int = 32):
    """C = sum_kb (A_q[:, kb] @ B_q[kb, :]) * sa[:, kb] * sb[kb, j-block].

    A_q, B_q hold FP8-rounded values (stored as float32 so the C backend can
    run them); sa is per (row, K-block) - DeepSeek-V3's 1x128 activation tiles -
    and sb per (K-block, N-block) - its 128x128 weight tiles. Each K-block's
    partial product is formed first and then promoted into the fp32
    accumulator with its scales, which is DeepSeek-V3's "promotion to CUDA
    cores" fix for limited tensor-core accumulation precision."""

    @T.prim_func
    def main(A: T.Tensor((M, K), "float32"), SA: T.Tensor((M, K // blk), "float32"),
             B: T.Tensor((K, N), "float32"), SB: T.Tensor((K // blk, N // blk), "float32"),
             C: T.Tensor((M, N), "float32")):
        with T.Kernel(N // blk, M // blk, threads=1) as (bx, by):
            acc = T.alloc_local((blk, blk), "float32")
            part = T.alloc_local((blk, blk), "float32")
            T.clear(acc)
            for kb in T.serial(K // blk):
                # TODO: clear part; part += A[by*blk+i, kb*blk+k] * B[kb*blk+k, bx*blk+j]; then acc += part * SA[row, kb] * SB[kb, bx]
                raise NotImplementedError("TODO(w12_tilelang): clear part; part += A[by*blk+i, kb*blk+k] * B[kb*blk+k, bx*blk+j]; then acc += part * SA[row, kb] * SB[kb, bx]")
            T.copy(acc, C[by * blk, bx * blk])

    return main


def quantize_blocks(A: torch.Tensor, B: torch.Tensor, blk: int):
    """FP8 E4M3 rounding with DeepSeek-style scales; returns float32 holders."""
    M, K = A.shape
    _, N = B.shape
    sa = A.reshape(M, K // blk, blk).abs().amax(-1).clamp(min=1e-12) / 448.0
    aq = (A.reshape(M, K // blk, blk) / sa[..., None]).to(torch.float8_e4m3fn).float().reshape(M, K)
    bt = B.reshape(K // blk, blk, N // blk, blk)
    sb = bt.abs().amax((1, 3)).clamp(min=1e-12) / 448.0
    bq = (bt / sb[:, None, :, None]).to(torch.float8_e4m3fn).float().reshape(K, N)
    return aq, sa.contiguous(), bq, sb.contiguous()


# ---------------------------------------------------------------- attention


def flash_attn_cpu(BH: int, S: int, D: int, causal: bool = True):
    """One program per (query row, batch*head): online softmax over keys."""
    scale = 1.0 / math.sqrt(D)

    @T.prim_func
    def main(Q: T.Tensor((BH, S, D), "float32"), K: T.Tensor((BH, S, D), "float32"),
             V: T.Tensor((BH, S, D), "float32"), O: T.Tensor((BH, S, D), "float32")):
        with T.Kernel(S, BH, threads=1) as (i, bh):
            m = T.alloc_local((1,), "float32")
            l = T.alloc_local((1,), "float32")
            s = T.alloc_local((1,), "float32")
            mn = T.alloc_local((1,), "float32")
            alpha = T.alloc_local((1,), "float32")
            p = T.alloc_local((1,), "float32")
            acc = T.alloc_local((D,), "float32")
            m[0] = -T.infinity("float32")
            l[0] = 0.0
            for d in T.serial(D):
                acc[d] = 0.0
            for j in T.serial(S):
                if (not causal) or j <= i:
                    # TODO: s = scale * dot(Q[bh,i,:], K[bh,j,:]); mn = max(m, s); alpha = exp(m-mn); p = exp(s-mn); l = l*alpha + p; acc = acc*alpha + p*V[bh,j,:]; m = mn
                    raise NotImplementedError("TODO(w12_tilelang): s = scale * dot(Q[bh,i,:], K[bh,j,:]); mn = max(m, s); alpha = exp(m-mn); p = exp(s-mn); l = l*alpha + p; acc = acc*alpha + p*V[bh,j,:]; m = mn")
            for d in T.serial(D):
                O[bh, i, d] = acc[d] / l[0]

    return main


# ---------------------------------------------------------------- driver


def run_gemm(A, B):
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    prog = gemm_cpu(M, N, K) if TARGET == "c" else gemm_gpu(M, N, K, dtype="float32" if A.dtype == torch.float32 else "float16")
    compile(prog)(A, B, C)
    return C


def run_blockscaled(A, B, blk=32):
    aq, sa, bq, sb = quantize_blocks(A, B, blk)
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty(M, N)
    compile(blockscaled_gemm_cpu(M, N, K, blk), "c")(aq, sa, bq, sb, C)
    ref = (aq.reshape(M, K // blk, blk) * sa[..., None]).reshape(M, K) @ \
          (bq.reshape(K // blk, blk, N // blk, blk) * sb[:, None, :, None]).reshape(K, N)
    return C, ref


def run_attention(q, k, v, causal=True):
    BH, S, D = q.shape
    o = torch.empty_like(q)
    compile(flash_attn_cpu(BH, S, D, causal), "c")(q.contiguous(), k.contiguous(), v.contiguous(), o)
    return o


if __name__ == "__main__":
    torch.manual_seed(0)
    A, B = torch.randn(64, 96), torch.randn(96, 64)
    print("gemm max |diff|:", (run_gemm(A, B) - A @ B).abs().max().item())
    C, ref = run_blockscaled(torch.randn(64, 128), torch.randn(128, 64))
    print("block-scaled FP8 GEMM vs dequantised reference:", (C - ref).abs().max().item())
    q, k, v = (torch.randn(4, 32, 16) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    print("attention max |diff|:", (run_attention(q, k, v) - ref).abs().max().item())
