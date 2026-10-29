"""Week 15: prefix reuse - vLLM-style block hashing vs SGLang-style radix tree.

Roadmap: "Compare vLLM prefix caching with SGLang RadixAttention on a
multi-turn chat trace and a RAG trace; record TTFT and throughput."

Two ways to find previously computed KV for a new prompt:

  BlockHashCache (vLLM automatic prefix caching)
    Only *full* blocks are cacheable. Block i's key = hash(key of block i-1,
    tokens of block i), so a key identifies the whole prefix up to that block.
    Lookup walks blocks from the start until a key misses.

  RadixCache (SGLang RadixAttention)
    A radix tree over token sequences. Edges hold token runs; a lookup returns
    the longest matching prefix at *token* granularity. Leaves with no active
    users are evicted LRU when capacity is exceeded.

Part 3 measures both on two synthetic traces and reports the fraction of
prefill tokens saved; part 4 plugs the block-hash cache into a real prefill on
the shared tiny GPT and measures TTFT. On GPU, gpu_bench.sh runs the same two
traces against vLLM and SGLang servers.
"""
from __future__ import annotations

import pathlib
import random
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch


# ------------------------------------------------------------------ 1. block hash


class BlockHashCache:
    def __init__(self, block_size: int, capacity_blocks: int):
        self.B = block_size
        self.cap = capacity_blocks
        self.store: dict[int, int] = {}  # key -> last-use tick (stand-in for a KV block)
        self.tick = 0

    def keys(self, tokens: list[int]) -> list[int]:
        """Chained hash key for every full block of `tokens`."""
        # >>> SOLUTION: for each full block i: h = hash((h_prev, tuple(block tokens))); collect keys
        out, h = [], None
        for i in range(len(tokens) // self.B):
            h = hash((h, tuple(tokens[i * self.B : (i + 1) * self.B])))
            out.append(h)
        return out
        # <<< SOLUTION

    def match(self, tokens: list[int]) -> int:
        """Number of prompt tokens whose KV can be reused."""
        n = 0
        self.tick += 1
        for k in self.keys(tokens):
            if k not in self.store:
                break
            self.store[k] = self.tick
            n += self.B
        return n

    def insert(self, tokens: list[int]):
        self.tick += 1
        for k in self.keys(tokens):
            self.store[k] = self.tick
        while len(self.store) > self.cap:
            del self.store[min(self.store, key=self.store.get)]  # LRU


# ------------------------------------------------------------------ 2. radix tree


@dataclass
class Node:
    tokens: tuple = ()
    children: dict = field(default_factory=dict)  # first token -> Node
    parent: "Node | None" = None
    last_use: int = 0
    ref: int = 0  # active requests pinning this node


class RadixCache:
    def __init__(self, capacity_tokens: int):
        self.root = Node()
        self.cap = capacity_tokens
        self.size = 0
        self.tick = 0

    def match(self, tokens: list[int]) -> int:
        """Length of the longest cached prefix of `tokens` (token granularity)."""
        self.tick += 1
        node, i = self.root, 0
        # >>> SOLUTION: walk children by tokens[i]; compare the edge run; stop at the first mismatch; update last_use on the path
        while i < len(tokens) and tokens[i] in node.children:
            child = node.children[tokens[i]]
            run = child.tokens
            j = 0
            while j < len(run) and i + j < len(tokens) and run[j] == tokens[i + j]:
                j += 1
            child.last_use = self.tick
            i += j
            if j < len(run):
                break
            node = child
        return i
        # <<< SOLUTION

    def insert(self, tokens: list[int]):
        self.tick += 1
        node, i = self.root, 0
        tokens = list(tokens)
        # >>> SOLUTION: walk like match; on a partial edge match split the edge into (common, rest); add a new leaf for the remaining tokens; size += new tokens
        while i < len(tokens):
            t = tokens[i]
            if t not in node.children:
                leaf = Node(tuple(tokens[i:]), parent=node, last_use=self.tick)
                node.children[t] = leaf
                self.size += len(leaf.tokens)
                break
            child = node.children[t]
            run = child.tokens
            j = 0
            while j < len(run) and i + j < len(tokens) and run[j] == tokens[i + j]:
                j += 1
            if j < len(run):  # split child at j
                mid = Node(run[:j], parent=node, last_use=self.tick)
                child.tokens, child.parent = run[j:], mid
                mid.children[run[j]] = child
                node.children[t] = mid
                child = mid
            child.last_use = self.tick
            node, i = child, i + j
        # <<< SOLUTION
        self._evict()

    def _evict(self):
        while self.size > self.cap:
            leaves = []
            stack = [self.root]
            while stack:
                n = stack.pop()
                if not n.children and n is not self.root and n.ref == 0:
                    leaves.append(n)
                stack.extend(n.children.values())
            if not leaves:
                return
            victim = min(leaves, key=lambda n: n.last_use)
            del victim.parent.children[victim.tokens[0]]
            self.size -= len(victim.tokens)


# ------------------------------------------------------------------ 3. traces


def multi_turn_trace(n_sessions=20, turns=6, sys_len=64, msg_len=24, reply_len=32, seed=0):
    """Chat: each turn's prompt = system + all previous turns + new message."""
    rnd = random.Random(seed)
    system = [rnd.randrange(256) for _ in range(sys_len)]
    sessions = [list(system) for _ in range(n_sessions)]
    trace = []
    for _ in range(turns):
        for s in range(n_sessions):
            sessions[s] += [rnd.randrange(256) for _ in range(msg_len)]
            trace.append(list(sessions[s]))
            sessions[s] += [rnd.randrange(256) for _ in range(reply_len)]  # model reply joins history
    return trace


def rag_trace(n=120, sys_len=64, n_docs=10, doc_len=200, q_len=20, seed=0):
    """RAG: system prompt + one of a few retrieved documents + a fresh question."""
    rnd = random.Random(seed)
    system = [rnd.randrange(256) for _ in range(sys_len)]
    docs = [[rnd.randrange(256) for _ in range(doc_len)] for _ in range(n_docs)]
    return [system + docs[rnd.randrange(n_docs)] + [rnd.randrange(256) for _ in range(q_len)] for _ in range(n)]


def no_reuse_trace(n=100, length=200, seed=0):
    rnd = random.Random(seed)
    return [[rnd.randrange(256) for _ in range(length)] for _ in range(n)]


def saved_fraction(cache, trace) -> float:
    total = saved = 0
    for prompt in trace:
        saved += cache.match(prompt)
        total += len(prompt)
        cache.insert(prompt)
    return saved / total


def compare(capacity_tokens: int = 50_000, block_size: int = 16) -> list[dict]:
    rows = []
    for name, tr in [("multi-turn", multi_turn_trace()), ("rag", rag_trace()), ("no-reuse", no_reuse_trace())]:
        rows.append({
            "trace": name,
            "block-hash saved": saved_fraction(BlockHashCache(block_size, capacity_tokens // block_size), tr),
            "radix saved": saved_fraction(RadixCache(capacity_tokens), tr),
        })
    return rows


# ------------------------------------------------------------------ 4. real TTFT on the tiny GPT


def ttft_experiment(trace, use_cache: bool, block_size: int = 16, d_model: int = 384) -> float:
    """Median time to first token. With the cache, only uncached tokens are prefilled:
    we run the model on the suffix with past KV taken from a dict keyed by block hash.

    The model must be big enough that prefill compute dominates hashing and
    tensor copies; with a toy model the bookkeeping costs as much as it saves,
    which is the same trade-off that shows up as offload overhead in production."""
    import statistics

    from common.tinygpt import GPT, GPTConfig

    torch.manual_seed(0)
    model = GPT(GPTConfig(seq_len=2048, n_layer=4, n_head=6, d_model=d_model)).eval()
    prefill_with_past(model, trace[0][:32], 0, None)  # warm-up: first call pays allocator/kernel setup
    cache = BlockHashCache(block_size, 10**9)
    kv: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}  # block key -> per-layer (k, v)
    times = []
    for prompt in trace:
        t0 = time.perf_counter()
        hit = cache.match(prompt) if use_cache else 0
        hit = min(hit, len(prompt) - 1)  # always compute at least the last token
        hit -= hit % block_size
        keys = cache.keys(prompt)
        past = None
        if hit:
            blocks = [kv[k] for k in keys[: hit // block_size]]
            past = [(torch.cat([b[l][0] for b in blocks], 2), torch.cat([b[l][1] for b in blocks], 2))
                    for l in range(len(model.blocks))]
        new_kv = prefill_with_past(model, prompt[hit:], hit, past)
        times.append(time.perf_counter() - t0)
        if use_cache:
            full = len(prompt) // block_size
            for bi in range(hit // block_size, full):
                s, e = bi * block_size - hit, (bi + 1) * block_size - hit
                kv[keys[bi]] = [(k[:, :, s:e], v[:, :, s:e]) for k, v in new_kv]
            cache.insert(prompt)
    return statistics.median(times)


@torch.no_grad()
def prefill_with_past(model, tokens: list[int], offset: int, past):
    """Run the suffix of a prompt given KV of the first `offset` tokens. Returns this suffix's KV."""
    import torch.nn.functional as F

    x = model.wte(torch.tensor(tokens))[None]
    pos = torch.arange(offset, offset + len(tokens))
    out_kv = []
    for li, blk in enumerate(model.blocks):
        q, k, v = blk.attn.split_qkv(blk.ln1(x), pos)
        out_kv.append((k, v))
        K = k if past is None else torch.cat([past[li][0], k], 2)
        V = v if past is None else torch.cat([past[li][1], v], 2)
        mask = torch.arange(offset + len(tokens))[None, :] <= pos[:, None]
        y = F.scaled_dot_product_attention(q, K, V, attn_mask=mask)
        x = x + blk.attn.proj(y.transpose(1, 2).reshape(1, len(tokens), -1))
        x = x + blk.mlp(blk.ln2(x))
    model.head(model.ln_f(x[:, -1:]))  # first-token logits
    return out_kv


if __name__ == "__main__":
    from common.bench import table

    print(table(compare(), ["trace", "block-hash saved", "radix saved"]))
    tr = multi_turn_trace(n_sessions=4, turns=4)
    print(f"multi-turn TTFT: no cache {ttft_experiment(tr, False)*1e3:.1f} ms, prefix cache {ttft_experiment(tr, True)*1e3:.1f} ms")
    nr = no_reuse_trace(n=16)
    print(f"no-reuse TTFT:   no cache {ttft_experiment(nr, False)*1e3:.1f} ms, prefix cache {ttft_experiment(nr, True)*1e3:.1f} ms")
