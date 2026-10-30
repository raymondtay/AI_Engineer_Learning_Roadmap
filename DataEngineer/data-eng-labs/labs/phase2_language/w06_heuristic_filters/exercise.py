# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 6: heuristic quality filters and PII redaction.

Roadmap: "Implement the MassiveText (Gopher) and C4 rules and FineWeb's three
custom rules; measure tokens removed by each; email/phone/IP redaction."

Thresholds are the defaults in datatrove 0.10.1 (read from its source):
  Gopher quality     50-100,000 words; mean word length 3-10; symbol/word <= 0.1;
                     bullet lines <= 90%; ellipsis lines <= 30%; >= 80% of words
                     contain a letter; >= 2 of [the be to of and that have with]
  Gopher repetition  duplicate lines <= 30% (by count) and <= 20% (by chars);
                     duplicate paragraphs likewise; top 2/3/4-gram char share
                     <= 0.20/0.18/0.16
  C4                 drop lorem ipsum, curly brackets; remove lines with
                     "javascript" or policy phrases (terminal-punctuation rule
                     left OFF, as FineWeb did: it removed ~30% of tokens)
  FineWeb            line-ending punctuation ratio <= 0.12 -> drop; duplicated
                     line chars >= 0.1 -> drop; lines < 30 chars >= 67% -> drop
Note: the FineWeb paper states 0.1 for the duplicated-line-character rule, but
datatrove's FineWebQualityFilter defaults char_duplicates_ratio to 0.01. We use
the paper's 0.1; try both and measure the difference.

These are English rules. Run them after language ID, on English only.
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table
from common.webgen import corpus

STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
END_PUNCT = (".", "?", "!", '"', "'")
POLICY = ("terms of use", "privacy policy", "cookie policy", "uses cookies", "use of cookies", "use cookies")


def words_of(text: str) -> list[str]:
    return text.split()


# ------------------------------------------------------------------ Gopher quality

def gopher_quality(text: str) -> str | None:
    """Return the name of the first failed rule, or None if the document passes."""
    words = words_of(text)
    n = len(words)
    lines = [l for l in text.split("\n") if l.strip()]
    # TODO: word-count range; mean word length; symbol ratio (# and ellipsis); bullets; ellipsis lines; alpha-word share; stop words
    raise NotImplementedError("TODO(w06_heuristic_filters): word-count range; mean word length; symbol ratio (# and ellipsis); bullets; ellipsis lines; alpha-word share; stop words")


# ------------------------------------------------------------------ Gopher repetition

def _dup_stats(items: list[str]) -> tuple[float, float]:
    """(fraction of items that are duplicates, fraction of chars in duplicate items)."""
    if not items:
        return 0.0, 0.0
    seen, dup_n, dup_chars = set(), 0, 0
    for it in items:
        if it in seen:
            dup_n += 1
            dup_chars += len(it)
        seen.add(it)
    return dup_n / len(items), dup_chars / max(1, sum(map(len, items)))


def top_ngram_char_frac(words: list[str], n: int) -> float:
    """Share of characters covered by the single most frequent n-gram."""
    grams = collections.Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    if not grams:
        return 0.0
    top, count = grams.most_common(1)[0]
    return sum(len(w) for w in top) * count / max(1, sum(len(w) for w in words))


def gopher_repetition(text: str) -> str | None:
    lines = [l for l in text.split("\n") if l.strip()]
    paras = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    # TODO: dup line frac > 0.3 or dup line char frac > 0.2; same for paragraphs; top 2/3/4-gram char frac > 0.20/0.18/0.16
    raise NotImplementedError("TODO(w06_heuristic_filters): dup line frac > 0.3 or dup line char frac > 0.2; same for paragraphs; top 2/3/4-gram char frac > 0.20/0.18/0.16")


# ------------------------------------------------------------------ C4

def c4(text: str) -> tuple[str | None, str]:
    """Document-level C4 rules plus line removal. Returns (failed rule or None, cleaned text)."""
    low = text.lower()
    # TODO: drop on 'lorem ipsum' or '{'; then remove lines containing 'javascript' or any POLICY phrase
    raise NotImplementedError("TODO(w06_heuristic_filters): drop on 'lorem ipsum' or '{'; then remove lines containing 'javascript' or any POLICY phrase")


# ------------------------------------------------------------------ FineWeb custom

def fineweb(text: str, char_dup: float = 0.1) -> str | None:
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return "fineweb_empty"
    # TODO: share of lines ending in END_PUNCT <= 0.12 -> fail; dup line chars / total chars >= char_dup -> fail; share of lines < 30 chars >= 0.67 -> fail
    raise NotImplementedError("TODO(w06_heuristic_filters): share of lines ending in END_PUNCT <= 0.12 -> fail; dup line chars / total chars >= char_dup -> fail; share of lines < 30 chars >= 0.67 -> fail")


# ------------------------------------------------------------------ PII

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"\+?\d{1,3}[-. ]?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")


def redact_pii(text: str) -> tuple[str, int]:
    """Replace emails, phone numbers and IPv4 addresses; return (text, replacements)."""
    # TODO: subn EMAIL -> <EMAIL>, PHONE -> <PHONE>, IPV4 -> <IP>; sum the counts
    raise NotImplementedError("TODO(w06_heuristic_filters): subn EMAIL -> <EMAIL>, PHONE -> <PHONE>, IPV4 -> <IP>; sum the counts")


# ------------------------------------------------------------------ pipeline

def run(docs: list[dict]) -> tuple[list[dict], dict]:
    """Apply C4 line cleaning, then all rules in order; attribute each removal to the first failing rule."""
    kept, removed_by = [], collections.Counter()
    tokens_removed = collections.Counter()
    for d in docs:
        reason, text = c4(d["text"])
        reason = reason or gopher_quality(text) or gopher_repetition(text) or fineweb(text)
        if reason:
            removed_by[reason] += 1
            tokens_removed[reason] += len(words_of(d["text"]))
            continue
        text, _ = redact_pii(text)
        kept.append({**d, "text": text})
    return kept, {"docs": removed_by, "tokens": tokens_removed}


def report(n: int = 3000) -> str:
    docs = [d for d in corpus(n, seed=1) if d["lang"] == "en"]
    kept, stats = run(docs)
    total_tok = sum(len(d["text"].split()) for d in docs)
    rows = [{"rule": r, "docs removed": c, "% tokens removed": 100 * stats["tokens"][r] / total_tok}
            for r, c in stats["docs"].most_common()]
    by_kind = collections.Counter(d["kind"] for d in docs)
    kept_kind = collections.Counter(d["kind"] for d in kept)
    kinds = [{"kind": k, "in": by_kind[k], "kept": kept_kind[k]} for k in by_kind]
    return table(rows, ["rule", "docs removed", "% tokens removed"]) + "\n\n" + table(kinds, ["kind", "in", "kept"])


if __name__ == "__main__":
    print(report())
