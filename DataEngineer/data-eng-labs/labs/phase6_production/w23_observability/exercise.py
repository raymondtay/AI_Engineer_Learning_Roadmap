# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 23: data-quality observability - profiles, drift tests, alerts.

Roadmap: "Profile every batch; detect drift (PSI, KS); alert on regressions
before they reach training. Inject failures and measure detection vs false alarms."

Silent data bugs are the expensive ones: an extractor update that starts
keeping navigation menus, a fetcher that truncates bodies, a language-ID model
swap. None of these crash anything. Each shows up as a shifted distribution.

  profile    per batch: doc length, share of lines ending in punctuation,
             non-alphabetic character share, language mix, filter pass rate
  PSI        Population Stability Index over binned feature distributions:
             sum (p_i - q_i) ln(p_i / q_i). Common rule of thumb (credit
             scoring, e.g. Siddiqi 2006): < 0.1 stable, 0.1-0.25 moderate, > 0.25 major.
  KS         two-sample Kolmogorov-Smirnov on raw values (scipy). With
             thousands of documents per batch the p-value flags *tiny* shifts;
             that is why the alert here uses an effect-size floor as well.
Calibration: run the detector on clean batches to measure the false alarm rate.

Stretch goal: the "spam +3 pts" injection is NOT caught by these distribution
monitors. Add a direct metric (share of docs rejected by your Week 6 filters,
with a binomial test against the reference rate) and show it catches it
without raising the false-alarm rate.
"""
from __future__ import annotations

import pathlib
import random
import re
import sys

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table
from common.webgen import corpus

FEATURES = ["log_words", "punct_lines", "non_alpha"]


def features(text: str) -> dict:
    words = text.split()
    lines = [l for l in text.splitlines() if l.strip()]
    return {"log_words": float(np.log1p(len(words))),
            "punct_lines": sum(l.rstrip().endswith((".", "!", "?")) for l in lines) / max(1, len(lines)),
            "non_alpha": sum(not (c.isalpha() or c.isspace()) for c in text) / max(1, len(text))}


def profile(docs: list[dict]) -> dict:
    F = [features(d["text"]) for d in docs]
    langs = {}
    for d in docs:
        langs[d["lang"]] = langs.get(d["lang"], 0) + 1
    return {"values": {f: np.array([x[f] for x in F]) for f in FEATURES},
            "lang": {k: v / len(docs) for k, v in langs.items()}}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI with quantile bins taken from the reference."""
    # TODO: interior edges = reference quantiles at 1/bins..(bins-1)/bins (deduplicated); bucket both samples with searchsorted; proportions floored at eps; sum (p-q)*ln(p/q)
    raise NotImplementedError("TODO(w23_observability): interior edges = reference quantiles at 1/bins..(bins-1)/bins (deduplicated); bucket both samples with searchsorted; proportions floored at eps; sum (p-q)*ln(p/q)")


def psi_categorical(ref: dict, cur: dict, eps: float = 1e-4) -> float:
    keys = set(ref) | set(cur)
    return float(sum((max(ref.get(k, 0), eps) - max(cur.get(k, 0), eps)) *
                     np.log(max(ref.get(k, 0), eps) / max(cur.get(k, 0), eps)) for k in keys))


def check(ref: dict, cur: dict, psi_alert: float = 0.25, ks_p: float = 1e-3, ks_floor: float = 0.1) -> list[str]:
    """Alert strings. A numeric feature alerts if PSI > psi_alert, or if KS is
    significant AND its statistic (max CDF gap) exceeds ks_floor."""
    alerts = []
    # TODO: for each feature compute psi and ks_2samp; append f"{feature}: PSI=..." / f"{feature}: KS=..." when a rule fires; then the language mix with psi_categorical
    raise NotImplementedError("TODO(w23_observability): for each feature compute psi and ks_2samp; append f'{feature}: PSI=...' / f'{feature}: KS=...' when a rule fires; then the language mix with psi_categorical")


# ------------------------------------------------------------------ failure injection

def clean_batch(seed: int, n: int = 1500) -> list[dict]:
    return corpus(n, seed=seed)


def broken_extractor(seed, n=1500):
    """Extractor regression: boilerplate share jumps from 8% to 35%."""
    return corpus(n, seed=seed, mix={"good": 0.45, "spam": 0.1, "boilerplate": 0.35, "code": 0.05, "lorem": 0.05})


def truncating_fetcher(seed, n=1500):
    """Fetcher bug: bodies cut at 1 KB for 40% of documents."""
    rnd = random.Random(seed)
    docs = corpus(n, seed=seed)
    for d in docs:
        if rnd.random() < 0.4:
            d["text"] = d["text"][:1024]
    return docs


def langid_swap(seed, n=1500):
    """A new language-ID model labels half of German as Dutch."""
    rnd = random.Random(seed)
    docs = corpus(n, seed=seed, langs={"en": 0.6, "de": 0.25, "fr": 0.08, "es": 0.07})
    for d in docs:
        if d["lang"] == "de" and rnd.random() < 0.5:
            d["lang"] = "nl"
    return docs


def subtle_spam(seed, n=1500):
    """Spam share 10% -> 13%: a real but small shift. Should it page someone?"""
    return corpus(n, seed=seed, mix={"good": 0.67, "spam": 0.13, "boilerplate": 0.08, "code": 0.07, "lorem": 0.05})


INJECTIONS = {"broken extractor": broken_extractor, "truncating fetcher": truncating_fetcher,
              "language-ID swap": langid_swap, "spam +3 pts (subtle)": subtle_spam}


def evaluate(trials: int = 10) -> list[dict]:
    ref = profile(clean_batch(seed=0, n=3000))
    rows = [{"batch": "clean (false alarms)", "detected": sum(bool(check(ref, profile(clean_batch(100 + s)))) for s in range(trials)) / trials, "example": ""}]
    for name, fn in INJECTIONS.items():
        res = [check(ref, profile(fn(200 + s))) for s in range(trials)]
        rows.append({"batch": name, "detected": sum(bool(r) for r in res) / trials,
                     "example": "; ".join(next((r for r in res if r), []))})
    return rows


def report() -> str:
    return ("Reference: 3000 clean docs. Each row: 10 batches of 1500 docs. Alert = PSI > 0.25, or KS p < 1e-3 with gap > 0.1.\n" +
            table(evaluate(), ["batch", "detected", "example"]))


if __name__ == "__main__":
    print(report())
