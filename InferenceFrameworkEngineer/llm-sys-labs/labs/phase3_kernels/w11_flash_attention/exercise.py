# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 11: FlashAttention forward and backward in Triton.

Roadmap: "Read the paper, then write FlashAttention forward and backward in
Triton. Add causal mask and variable-length (packed) sequences."
Gate (GPU): within 1.2x of the official FlashAttention-2 at 4K sequence length.

Layout: packed "varlen" (total_tokens, n_heads, head_dim) plus cu_seqlens, the
same interface as flash_attn_varlen_func. A padded (B, H, T, D) batch is just
the special case of equal lengths; `attention()` converts for you.

Algorithm (Dao et al. 2022, FA2 2023):
  forward  - for each query block, stream K/V blocks through on-chip memory,
             keep a running max m and running sum l (online softmax), never
             materialise the T x T score matrix. Save LSE = m + log(l).
  backward - recompute P = exp(S - LSE) block by block. Two kernels:
             dK/dV loops over query blocks for a fixed K/V block;
             dQ loops over K/V blocks for a fixed query block. No atomics.
             delta = rowsum(dO * O) is precomputed.
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import common  # noqa: F401
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(Q, K, V, O, LSE, CU, sm_scale, st, sh, H,
         D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m, pid_bh = tl.program_id(0), tl.program_id(1)
    b, h = pid_bh // H, pid_bh % H
    start = tl.load(CU + b)
    L = tl.load(CU + b + 1) - start
    if pid_m * BM >= L:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + (start + offs_m)[:, None] * st + h * sh + offs_d[None, :], mask=offs_m[:, None] < L, other=0.0)
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    end_n = tl.minimum(L, (pid_m + 1) * BM) if CAUSAL else L
    for n0 in range(0, end_n, BN):
        cols = n0 + offs_n
        kv_off = (start + cols)[:, None] * st + h * sh + offs_d[None, :]
        k = tl.load(K + kv_off, mask=cols[:, None] < L, other=0.0)
        v = tl.load(V + kv_off, mask=cols[:, None] < L, other=0.0)
        # TODO: s = q k^T * scale; mask (cols < L, and cols <= rows if causal) to -inf; online-softmax update of m_i, l_i, acc
        raise NotImplementedError("TODO(w11_flash_attention): s = q k^T * scale; mask (cols < L, and cols <= rows if causal) to -inf; online-softmax update of m_i, l_i, acc")
    row_ok = offs_m < L
    tl.store(O + (start + offs_m)[:, None] * st + h * sh + offs_d[None, :], (acc / l_i[:, None]).to(O.dtype.element_ty),
             mask=row_ok[:, None])
    # LSE is laid out (total_tokens, H), so row r of head h lives at r * H + h.
    tl.store(LSE + (start + offs_m) * H + h, m_i + tl.log(l_i), mask=row_ok)


@triton.jit
def _bwd_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV, CU, sm_scale, st, sh, H,
              D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr):
    pid_n, pid_bh = tl.program_id(0), tl.program_id(1)
    b, h = pid_bh // H, pid_bh % H
    start = tl.load(CU + b)
    L = tl.load(CU + b + 1) - start
    if pid_n * BN >= L:
        return
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    kv_off = (start + offs_n)[:, None] * st + h * sh + offs_d[None, :]
    k = tl.load(K + kv_off, mask=offs_n[:, None] < L, other=0.0)
    v = tl.load(V + kv_off, mask=offs_n[:, None] < L, other=0.0)
    dk = tl.zeros((BN, D), tl.float32)
    dv = tl.zeros((BN, D), tl.float32)
    m_start = (pid_n * BN // BM) * BM if CAUSAL else 0
    for m0 in range(m_start, L, BM):
        rows = m0 + tl.arange(0, BM)
        q_off = (start + rows)[:, None] * st + h * sh + offs_d[None, :]
        q = tl.load(Q + q_off, mask=rows[:, None] < L, other=0.0)
        do = tl.load(DO + q_off, mask=rows[:, None] < L, other=0.0)
        lse = tl.load(LSE + (start + rows) * H + h, mask=rows < L, other=0.0)
        delta = tl.load(DELTA + (start + rows) * H + h, mask=rows < L, other=0.0)
        # TODO: recompute p = exp(s - lse) with masking (zero outside); dv += p^T do; dp = do v^T; ds = p*(dp - delta); dk += ds^T q * scale
        raise NotImplementedError("TODO(w11_flash_attention): recompute p = exp(s - lse) with masking (zero outside); dv += p^T do; dp = do v^T; ds = p*(dp - delta); dk += ds^T q * scale")
    tl.store(DK + kv_off, dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < L)
    tl.store(DV + kv_off, dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < L)


@triton.jit
def _bwd_dq(Q, K, V, DO, LSE, DELTA, DQ, CU, sm_scale, st, sh, H,
            D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m, pid_bh = tl.program_id(0), tl.program_id(1)
    b, h = pid_bh // H, pid_bh % H
    start = tl.load(CU + b)
    L = tl.load(CU + b + 1) - start
    if pid_m * BM >= L:
        return
    rows = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q_off = (start + rows)[:, None] * st + h * sh + offs_d[None, :]
    q = tl.load(Q + q_off, mask=rows[:, None] < L, other=0.0)
    do = tl.load(DO + q_off, mask=rows[:, None] < L, other=0.0)
    lse = tl.load(LSE + (start + rows) * H + h, mask=rows < L, other=0.0)
    delta = tl.load(DELTA + (start + rows) * H + h, mask=rows < L, other=0.0)
    dq = tl.zeros((BM, D), tl.float32)
    end_n = tl.minimum(L, (pid_m + 1) * BM) if CAUSAL else L
    for n0 in range(0, end_n, BN):
        cols = n0 + tl.arange(0, BN)
        kv_off = (start + cols)[:, None] * st + h * sh + offs_d[None, :]
        k = tl.load(K + kv_off, mask=cols[:, None] < L, other=0.0)
        v = tl.load(V + kv_off, mask=cols[:, None] < L, other=0.0)
        # TODO: same p and ds as dk/dv; dq += ds k * scale
        raise NotImplementedError("TODO(w11_flash_attention): same p and ds as dk/dv; dq += ds k * scale")
    tl.store(DQ + q_off, dq.to(DQ.dtype.element_ty), mask=rows[:, None] < L)


BM = BN = 32


class FlashAttnVarlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_len: int, causal: bool):
        total, H, D = q.shape
        assert D in (16, 32, 64, 128) and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        o = torch.empty_like(q)
        lse = torch.empty(total, H, device=q.device, dtype=torch.float32)
        nb = cu_seqlens.numel() - 1
        scale = 1.0 / math.sqrt(D)
        grid = (triton.cdiv(max_len, BM), nb * H)
        _fwd[grid](q, k, v, o, lse, cu_seqlens, scale, q.stride(0), q.stride(1), H, D=D, BM=BM, BN=BN, CAUSAL=causal)
        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens)
        ctx.max_len, ctx.causal, ctx.scale = max_len, causal, scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, cu = ctx.saved_tensors
        do = do.contiguous()
        total, H, D = q.shape
        delta = (do.float() * o.float()).sum(-1)  # (total, H)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        nb = cu.numel() - 1
        args = (ctx.scale, q.stride(0), q.stride(1), H)
        kw = dict(D=D, BM=BM, BN=BN, CAUSAL=ctx.causal)
        _bwd_dkdv[(triton.cdiv(ctx.max_len, BN), nb * H)](q, k, v, do, lse, delta, dk, dv, cu, *args, **kw)
        _bwd_dq[(triton.cdiv(ctx.max_len, BM), nb * H)](q, k, v, do, lse, delta, dq, cu, *args, **kw)
        return dq, dk, dv, None, None, None


def flash_attn_varlen(q, k, v, cu_seqlens, max_len, causal=True):
    return FlashAttnVarlen.apply(q, k, v, cu_seqlens, max_len, causal)


def attention(q, k, v, causal=True):
    """Padded (B, H, T, D) convenience wrapper."""
    B, H, T, D = q.shape
    pack = lambda t: t.transpose(1, 2).reshape(B * T, H, D).contiguous()
    cu = torch.arange(0, (B + 1) * T, T, device=q.device, dtype=torch.int32)
    o = flash_attn_varlen(pack(q), pack(k), pack(v), cu, T, causal)
    return o.view(B, T, H, D).transpose(1, 2)


def reference_varlen(q, k, v, cu, causal=True):
    """Loop over sequences with torch SDPA."""
    outs = []
    for i in range(cu.numel() - 1):
        s, e = cu[i].item(), cu[i + 1].item()
        t = lambda x: x[s:e].transpose(0, 1)[None]
        outs.append(torch.nn.functional.scaled_dot_product_attention(t(q), t(k), t(v), is_causal=causal)[0].transpose(0, 1))
    return torch.cat(outs)


def benchmark_gpu(T: int = 4096, B: int = 4, H: int = 16, D: int = 64) -> str:
    """Gate: ours vs flash_attn (if installed) vs SDPA at 4K tokens, fp16, fwd+bwd."""
    from common.bench import bench

    q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3))
    g = torch.randn_like(q)
    res = {"ours": bench(lambda: torch.autograd.grad(attention(q, k, v), (q, k, v), g))}
    res["sdpa"] = bench(lambda: torch.autograd.grad(
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True), (q, k, v), g))
    try:
        from flash_attn import flash_attn_func

        qt, kt, vt = (x.detach().transpose(1, 2).contiguous().requires_grad_() for x in (q, k, v))
        gt = g.transpose(1, 2).contiguous()
        res["flash_attn"] = bench(lambda: torch.autograd.grad(flash_attn_func(qt, kt, vt, causal=True), (qt, kt, vt), gt))
    except ImportError:
        pass
    base = res.get("flash_attn", res["sdpa"])
    return "  ".join(f"{k}={v*1e3:.2f}ms" for k, v in res.items()) + f"   ours/{'fa2' if 'flash_attn' in res else 'sdpa'} = {res['ours']/base:.2f}x (gate <= 1.2x)"


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(benchmark_gpu())
    else:
        torch.manual_seed(0)
        cu = torch.tensor([0, 37, 100, 164], dtype=torch.int32)
        q, k, v = (torch.randn(164, 2, 32) for _ in range(3))
        err = (flash_attn_varlen(q, k, v, cu, 64) - reference_varlen(q, k, v, cu)).abs().max()
        print(f"varlen forward max |diff| vs SDPA loop: {err:.2e}  (CPU interpreter; run on GPU for the speed gate)")
