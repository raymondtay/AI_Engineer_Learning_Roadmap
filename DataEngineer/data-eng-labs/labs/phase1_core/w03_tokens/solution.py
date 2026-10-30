"""Week 3: tokens as the unit of account - byte-level BPE from scratch.

Roadmap: "Train a small BPE tokenizer; measure tokens per byte by language and
by document type. Every later gate reports 'tokens kept'."

Why a data engineer cares: budgets, mixtures and dedup thresholds are all set
in tokens, and the same bytes cost very different numbers of tokens depending
on language and document type. A tokenizer trained mostly on English makes
other languages more expensive per byte - a mixture decision in disguise.

Algorithm (GPT-2 style, simplified):
  1. pre-tokenise: split text into words with their leading space, so merges
     never cross word boundaries
  2. start from bytes (256 symbols); count adjacent pairs weighted by word frequency
  3. repeatedly merge the most frequent pair into a new symbol
  4. encode = apply merges in the order they were learned
"""
from __future__ import annotations

import collections
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table
from common.webgen import code_doc, good_doc, spam_doc

PRETOKEN = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+", re.UNICODE)


def pretokenize(text: str) -> list[bytes]:
    return [m.group().encode("utf-8") for m in PRETOKEN.finditer(text)]


def train(texts: list[str], vocab_size: int) -> list[tuple[int, int]]:
    """Return merges [(a, b), ...]; merge i creates symbol 256 + i."""
    words = collections.Counter(w for t in texts for w in pretokenize(t))
    seqs = {w: list(w) for w in words}
    merges: list[tuple[int, int]] = []
    # >>> SOLUTION: loop until vocab_size: count pairs over seqs weighted by word count; pick most frequent (ties -> smallest pair); replace in every seq
    while 256 + len(merges) < vocab_size:
        pairs = collections.Counter()
        for w, s in seqs.items():
            c = words[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        new = 256 + len(merges)
        merges.append(best)
        for w, s in seqs.items():
            i, out = 0, []
            while i < len(s):
                if i + 1 < len(s) and (s[i], s[i + 1]) == best:
                    out.append(new)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[w] = out
    # <<< SOLUTION
    return merges


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    rank = {p: i for i, p in enumerate(merges)}
    out = []
    for w in pretokenize(text):
        s = list(w)
        # >>> SOLUTION: repeatedly find the adjacent pair with the lowest merge rank and merge it, until none applies
        while len(s) > 1:
            best = min(((rank.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(s, s[1:]))))
            if best[0] == 1 << 30:
                break
            r, i = best
            s = s[:i] + [256 + r] + s[i + 2:]
        # <<< SOLUTION
        out.extend(s)
    return out


def decode(ids: list[int], merges: list[tuple[int, int]]) -> str:
    table_ = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        table_[256 + i] = table_[a] + table_[b]
    return b"".join(table_[i] for i in ids).decode("utf-8", errors="replace")


def tokens_per_byte(texts: list[str], merges) -> float:
    n_tok = sum(len(encode(t, merges)) for t in texts)
    n_bytes = sum(len(t.encode()) for t in texts)
    return n_tok / n_bytes


def report(vocab: int = 800, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    train_texts = [good_doc(rnd, "en") for _ in range(150)] + [good_doc(rnd, l) for l in ("de", "fr", "es") for _ in range(10)]
    merges = train(train_texts, vocab)
    rnd = random.Random(seed + 1)
    eval_sets = {
        "english prose": [good_doc(rnd, "en") for _ in range(20)],
        "german prose": [good_doc(rnd, "de") for _ in range(20)],
        "french prose": [good_doc(rnd, "fr") for _ in range(20)],
        "code": [code_doc(rnd) for _ in range(20)],
        "spam": [spam_doc(rnd) for _ in range(20)],
    }
    return [{"set": k, "tokens/byte": tokens_per_byte(v, merges)} for k, v in eval_sets.items()]


if __name__ == "__main__":
    print(table(report(), ["set", "tokens/byte"]))
