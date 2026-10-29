# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 7: long context - context parallel (ring attention) and Ulysses.

Roadmap: "Sequence parallel, context parallel (ring attention), Ulysses. Train
at 32K tokens; budget activation memory per GPU."

Both schemes shard the *sequence* across ranks, so per-GPU activation memory
falls by the CP degree.

  Ring attention (Liu et al. 2023; Megatron-Core "context parallel"):
    each rank keeps its Q chunk and passes K/V chunks around a ring, merging
    partial results with the online-softmax rescaling FlashAttention uses.
    Communication overlaps with compute; works for any head count.

  Ulysses (DeepSpeed, Jacobs et al. 2023):
    one all-to-all turns "my tokens, all heads" into "all tokens, my heads",
    run ordinary attention, then all-to-all back. Needs n_head % cp == 0.

Causal masking makes naive ring attention load-imbalanced: rank 0 skips
almost every block. Production code uses a zigzag split (chunks r and 2c-1-r);
that is the stretch exercise at the bottom.
"""
from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist
import torch.nn.functional as F


def activation_bytes_per_layer(s: int, b: int, h: int, a: int, tp: int = 1, cp: int = 1, flash: bool = True,
                               bytes_per_el: int = 2) -> float:
    """Per-GPU activation memory of one transformer layer, 16-bit, no recompute.

    Korthikanti et al. 2022 ("Reducing Activation Recomputation in Large
    Transformer Models"): s*b*h*(34 + 5*a*s/h) bytes with tensor + sequence
    parallel dividing everything by t. FlashAttention removes the 5as/h term
    (the s x s score matrix is never stored). Context parallel divides s by cp.
    The 34 assumes 2-byte elements; we scale by bytes_per_el / 2."""
    # TODO: s_local = s/cp; base = s_local*b*h*34; if not flash add s_local*b*h*5*a*s_local/h; divide by tp; scale bytes
    raise NotImplementedError("TODO(w07_context_parallel): s_local = s/cp; base = s_local*b*h*34; if not flash add s_local*b*h*5*a*s_local/h; divide by tp; scale bytes")


def _block_update(q, k, v, m, l, acc, mask):
    """Merge one K/V block into the running (max, sum, output) state."""
    # TODO: scores scaled + masked; m_new = max(m, rowmax); rescale l and acc by exp(m - m_new); add exp(scores - m_new)
    raise NotImplementedError("TODO(w07_context_parallel): scores scaled + masked; m_new = max(m, rowmax); rescale l and acc by exp(m - m_new); add exp(scores - m_new)")


def ring_attention(q, k, v, rank: int, world: int):
    """Causal attention for this rank's chunk. q, k, v: (B, H, Tc, D) local chunks,
    chunk index == rank. Returns (B, H, Tc, D)."""
    B, H, Tc, D = q.shape
    m = torch.full((B, H, Tc, 1), float("-inf"), dtype=torch.float32)
    l = torch.zeros(B, H, Tc, 1, dtype=torch.float32)
    acc = torch.zeros(B, H, Tc, D, dtype=torch.float32)
    q32 = q.float()
    k_cur, v_cur = k.contiguous().float(), v.contiguous().float()
    diag = torch.ones(Tc, Tc, dtype=torch.bool).tril()
    for step in range(world):
        src = (rank - step) % world  # which chunk's K/V we hold now
        # TODO: start the ring send/recv for the next step, update state if src <= rank (diag mask when equal), then wait
        raise NotImplementedError("TODO(w07_context_parallel): start the ring send/recv for the next step, update state if src <= rank (diag mask when equal), then wait")
    return (acc / l).to(q.dtype)


def ulysses_attention(q, k, v, rank: int, world: int):
    """q, k, v: (B, H, Tc, D) sequence-sharded. All-to-all to head-sharded,
    full causal attention, all-to-all back."""
    B, H, Tc, D = q.shape
    assert H % world == 0
    hl = H // world

    def seq_to_head(t):
        # (B,H,Tc,D) -> send head group i to rank i -> receive every rank's Tc chunk of my heads
        # TODO: reshape to (world, B, hl, Tc, D), all_to_all_single, then concat the world chunks along T
        raise NotImplementedError("TODO(w07_context_parallel): reshape to (world, B, hl, Tc, D), all_to_all_single, then concat the world chunks along T")

    def head_to_seq(t):
        # TODO: inverse: split T into world chunks, all_to_all_single, put head groups back in order
        raise NotImplementedError("TODO(w07_context_parallel): inverse: split T into world chunks, all_to_all_single, put head groups back in order")

    o = F.scaled_dot_product_attention(seq_to_head(q), seq_to_head(k), seq_to_head(v), is_causal=True)
    return head_to_seq(o)


def _inputs(B=2, H=4, T=32, D=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, H, T, D, generator=g) for _ in range(3)]


def cp_check(rank: int, world: int, mode: str):
    q, k, v = _inputs()
    T = q.shape[2]
    Tc = T // world
    sl = slice(rank * Tc, (rank + 1) * Tc)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)[:, :, sl]
    fn = ring_attention if mode == "ring" else ulysses_attention
    out = fn(q[:, :, sl].contiguous(), k[:, :, sl].contiguous(), v[:, :, sl].contiguous(), rank, world)
    return (out - ref).abs().max().item()


# Stretch (not tested): zigzag load balancing. Split the sequence into 2*cp
# chunks and give rank r chunks r and 2*cp-1-r, so every rank does the same
# amount of causal work. Count blocks computed per rank with and without it.

if __name__ == "__main__":
    from common import dist as ldist

    for mode in ("ring", "ulysses"):
        print(mode, "max |diff| per rank:", ldist.run(cp_check, 4, mode))
    gb = lambda x: x / 2**30
    for cp in (1, 4, 8):
        print(f"32K tokens, 7B-ish layer (h=4096, a=32), tp=1, cp={cp}: "
              f"{gb(activation_bytes_per_layer(32768, 1, 4096, 32, cp=cp)):.2f} GiB/layer (flash)")
