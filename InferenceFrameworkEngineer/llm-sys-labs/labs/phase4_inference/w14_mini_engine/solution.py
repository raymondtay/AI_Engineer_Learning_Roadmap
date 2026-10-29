"""Week 14: a mini inference engine - paged KV cache, continuous batching, chunked prefill.

Roadmap: "Write a mini engine: continuous batching, paged KV cache, chunked
prefill. Then trace a request through vLLM's scheduler and block manager."
Gate (GPU): throughput within 2x of vLLM on the same GPU and model.

Design (the vLLM v1 shape, minus everything that is not essential):
  BlockAllocator  - fixed pool of KV blocks, free list, per-sequence block tables
  PagedKVCache    - (layers, num_blocks * block_size, heads, head_dim) K and V;
                    a token at position p of sequence s lives in slot
                    table[s][p // B] * B + p % B
  Scheduler       - each step has a token budget. Running sequences decode
                    (1 token each); waiting sequences prefill in chunks that fit
                    the remaining budget. If blocks run out, the newest running
                    sequence is preempted and recomputed later (vLLM's policy).
  Engine.step     - ONE ragged forward pass over all scheduled tokens: the
                    linear layers see every token at once; attention reads each
                    sequence's KV through its block table.

Uses the shared tiny GPT (common/tinygpt.py) so the weights are known-good.
"""
from __future__ import annotations

import itertools
import pathlib
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F

from common.device import device
from common.tinygpt import GPT, GPTConfig


# ------------------------------------------------------------------ memory


class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.free = list(range(num_blocks - 1, -1, -1))
        self.num_blocks = num_blocks

    def alloc(self) -> int:
        if not self.free:
            raise OutOfBlocks
        return self.free.pop()

    def release(self, blocks: list[int]):
        self.free.extend(reversed(blocks))

    @property
    def num_free(self) -> int:
        return len(self.free)


class PagedKVCache:
    def __init__(self, cfg: GPTConfig, num_blocks: int, block_size: int, dev):
        shape = (cfg.n_layer, num_blocks * block_size, cfg.n_head, cfg.head_dim)
        self.k = torch.zeros(shape, device=dev)
        self.v = torch.zeros(shape, device=dev)
        self.block_size = block_size

    def slots(self, table: list[int], positions: torch.Tensor) -> torch.Tensor:
        """Flat cache slot for each position of a sequence."""
        # >>> SOLUTION: slot = table[pos // block_size] * block_size + pos % block_size
        t = torch.tensor(table, device=positions.device)
        return t[positions // self.block_size] * self.block_size + positions % self.block_size
        # <<< SOLUTION


# ------------------------------------------------------------------ requests


@dataclass
class Sequence:
    sid: int
    prompt: list[int]
    max_new: int
    out: list[int] = field(default_factory=list)
    table: list[int] = field(default_factory=list)
    n_cached: int = 0           # tokens whose K/V are in the cache
    prefill_len: int = -1       # tokens to (re)compute before sampling; set on admit/preempt
    arrival: float = 0.0
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.out

    @property
    def done(self) -> bool:
        return len(self.out) >= self.max_new

    def __post_init__(self):
        if self.prefill_len < 0:
            self.prefill_len = len(self.prompt)

    @property
    def in_prefill(self) -> bool:
        return self.n_cached < self.prefill_len


@dataclass
class Chunk:
    seq: Sequence
    start: int  # first position computed this step
    n: int      # number of tokens computed this step


# ------------------------------------------------------------------ engine


class Engine:
    def __init__(self, model: GPT, num_blocks: int = 64, block_size: int = 8, token_budget: int = 64):
        self.model = model.eval()
        self.cfg = model.cfg
        self.dev = next(model.parameters()).device
        self.alloc = BlockAllocator(num_blocks)
        self.cache = PagedKVCache(self.cfg, num_blocks, block_size, self.dev)
        self.B = block_size
        self.budget = token_budget
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self._ids = itertools.count()
        self.stats = {"steps": 0, "tokens": 0, "preemptions": 0}

    # ---------------- public API
    def add_request(self, prompt: list[int], max_new: int) -> int:
        s = Sequence(next(self._ids), list(prompt), max_new, arrival=time.perf_counter())
        self.waiting.append(s)
        return s.sid

    def run(self) -> dict[int, list[int]]:
        while self.waiting or self.running:
            self.step()
        return {s.sid: s.out for s in sorted(self.finished, key=lambda s: s.sid)}

    # ---------------- scheduling
    def _ensure_blocks(self, s: Sequence, upto: int):
        """Make sure s has blocks for positions [0, upto)."""
        while len(s.table) * self.B < upto:
            s.table.append(self.alloc.alloc())

    def _preempt(self, s: Sequence):
        """Recompute-style preemption: drop KV, go back to the waiting queue.
        When re-admitted, all current tokens (prompt + generated so far) are
        prefilled again, then decoding continues; outputs are unchanged."""
        self.alloc.release(s.table)
        s.table, s.n_cached = [], 0
        s.prefill_len = len(s.tokens)
        self.running.remove(s)
        # Invariant: only the queue head may be partially prefilled (and hold
        # blocks). Putting the victim in front of it would strand those blocks
        # and can deadlock the pool, so it goes right behind a partial head.
        head_partial = bool(self.waiting) and self.waiting[0].n_cached > 0
        self.waiting.insert(1 if head_partial else 0, s)
        self.stats["preemptions"] += 1

    def schedule(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        budget = self.budget
        # >>> SOLUTION: 1) each running seq decodes 1 token (needs a block for position len(tokens)-1; preempt newest running seq on OutOfBlocks). 2) then admit waiting seqs: prefill min(prefill_len - n_cached, budget) tokens, allocating blocks; stop when budget or blocks run out
        for s in list(self.running):
            if budget == 0:
                break
            if s not in self.running:  # preempted earlier in this loop
                continue
            pos = len(s.tokens) - 1
            while True:
                try:
                    self._ensure_blocks(s, pos + 1)
                    break
                except OutOfBlocks:
                    victim = self.running[-1]
                    self._preempt(victim)
                    chunks = [c for c in chunks if c.seq is not victim]
                    if victim is s:
                        break
            if s in self.running:
                chunks.append(Chunk(s, pos, 1))
                budget -= 1
        while self.waiting and budget > 0:
            s = self.waiting[0]
            # Size the chunk to the blocks we can actually get, so a failed
            # allocation never leaves half-allocated blocks behind.
            capacity = (len(s.table) + self.alloc.num_free) * self.B - s.n_cached
            n = min(s.prefill_len - s.n_cached, budget, capacity)
            if n <= 0:
                break
            self._ensure_blocks(s, s.n_cached + n)
            chunks.append(Chunk(s, s.n_cached, n))
            budget -= n
            if s.n_cached + n == s.prefill_len:
                self.waiting.pop(0)
                self.running.append(s)
            else:
                break  # chunked prefill: this sequence used the rest of the budget
        # <<< SOLUTION
        return chunks

    # ---------------- model execution
    @torch.no_grad()
    def forward(self, chunks: list[Chunk]) -> torch.Tensor:
        """Ragged forward over all scheduled tokens. Returns logits of each chunk's last token."""
        toks, pos, slots = [], [], []
        for c in chunks:
            p = torch.arange(c.start, c.start + c.n, device=self.dev)
            toks.append(torch.tensor(c.seq.tokens[c.start : c.start + c.n], device=self.dev))
            pos.append(p)
            slots.append(self.cache.slots(c.seq.table, p))
        toks, pos, slots = torch.cat(toks), torch.cat(pos), torch.cat(slots)
        x = self.model.wte(toks)[None]  # (1, N, C)
        for li, blk in enumerate(self.model.blocks):
            h = blk.ln1(x)
            q, k, v = blk.attn.split_qkv(h, pos[None])  # (1, H, N, D)
            # >>> SOLUTION: write this step's K/V into the cache at `slots`; then for each chunk gather its K/V for positions [0, start+n) and run causal attention with offset
            self.cache.k[li, slots] = k[0].transpose(0, 1)
            self.cache.v[li, slots] = v[0].transpose(0, 1)
            outs, off = [], 0
            for c in chunks:
                L = c.start + c.n
                kv_slots = self.cache.slots(c.seq.table, torch.arange(L, device=self.dev))
                K = self.cache.k[li, kv_slots].transpose(0, 1)  # (H, L, D)
                V = self.cache.v[li, kv_slots].transpose(0, 1)
                qc = q[0, :, off : off + c.n]                   # (H, n, D)
                qpos = torch.arange(c.start, L, device=self.dev)
                mask = torch.arange(L, device=self.dev)[None, :] <= qpos[:, None]
                outs.append(F.scaled_dot_product_attention(qc, K, V, attn_mask=mask))
                off += c.n
            y = torch.cat(outs, dim=1)  # (H, N, D)
            # <<< SOLUTION
            x = x + blk.attn.proj(y.transpose(0, 1).reshape(1, -1, self.cfg.d_model))
            x = x + blk.mlp(blk.ln2(x))
        logits = self.model.head(self.model.ln_f(x))[0]
        last = torch.cumsum(torch.tensor([c.n for c in chunks], device=self.dev), 0) - 1
        return logits[last]

    def step(self):
        chunks = self.schedule()
        if not chunks:
            raise RuntimeError("scheduler made no progress: pool too small for one sequence?")
        logits = self.forward(chunks)
        now = time.perf_counter()
        for c, lg in zip(chunks, logits):
            s = c.seq
            s.n_cached = c.start + c.n
            if s.in_prefill:
                continue  # partial prefill chunk: no token sampled yet
            s.out.append(int(lg.argmax()))
            if s.first_token_time is None:
                s.first_token_time = now
            if s.done:
                s.finish_time = now
                self.running.remove(s)
                self.alloc.release(s.table)
                s.table = []
                self.finished.append(s)
        self.stats["steps"] += 1
        self.stats["tokens"] += sum(c.n for c in chunks)


# ------------------------------------------------------------------ reference & bench


@torch.no_grad()
def greedy_reference(model: GPT, prompt: list[int], n: int) -> list[int]:
    """No cache: re-run the full sequence each token. Slow but obviously right."""
    toks = torch.tensor(prompt, device=next(model.parameters()).device)[None]
    out = []
    for _ in range(n):
        logits, _ = model(toks)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        toks = torch.cat([toks, torch.tensor([[nxt]], device=toks.device)], 1)
    return out


def make_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(GPTConfig(seq_len=256, n_layer=2, n_head=4, d_model=64)).to(device())


def benchmark(n_requests: int = 16, prompt_len: int = 48, gen: int = 32) -> dict:
    model = make_model()
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(0, 256, (prompt_len,), generator=g).tolist() for _ in range(n_requests)]
    t0 = time.perf_counter()
    for p in prompts:
        greedy_reference(model, p, gen)
    t_naive = time.perf_counter() - t0
    eng = Engine(model, num_blocks=n_requests * ((prompt_len + gen) // 8 + 2), block_size=8, token_budget=256)
    for p in prompts:
        eng.add_request(p, gen)
    t0 = time.perf_counter()
    eng.run()
    t_eng = time.perf_counter() - t0
    toks = n_requests * gen
    return {"naive tok/s": toks / t_naive, "engine tok/s": toks / t_eng, "speedup": t_naive / t_eng, **eng.stats}


if __name__ == "__main__":
    print(benchmark())
