"""Week 5: main-text extraction and language identification.

Roadmap: "HTML to main text (Trafilatura vs your own boilerplate removal);
language ID; read WET vs re-extracting from WARC."

Extraction: Common Crawl's WET files are a naive text dump of the whole page,
navigation and footers included. Pipelines such as FineWeb re-extract from WARC
HTML with Trafilatura. The core idea fits in a page: walk the DOM, drop
non-content tags, then keep text blocks that are long and not mostly links.

Language ID: FineWeb keeps documents with a fastText English score >= 0.65.
fastText's idea in miniature is a classifier over character n-grams; here a
multinomial naive Bayes over 1-3-grams, which is enough to see why short
texts are unreliable and why a probability threshold matters.

Optional comparison: `pip install trafilatura` and call trafilatura.extract().
"""
from __future__ import annotations

import collections
import math
import pathlib
import random
import re
import sys
from html.parser import HTMLParser

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table
from common.webgen import good_doc, sentence, to_html

DROP_TAGS = {"script", "style", "nav", "footer", "aside", "header", "noscript", "form"}
BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "td", "article", "section", "br"}


class _Blocks(HTMLParser):
    """Collect text blocks with their link-text length."""

    def __init__(self):
        super().__init__()
        self.skip = 0
        self.in_a = 0
        self.blocks: list[dict] = []
        self.cur = {"text": [], "link_chars": 0}

    def _flush(self):
        text = " ".join("".join(self.cur["text"]).split())
        if text:
            self.blocks.append({"text": text, "link_chars": self.cur["link_chars"]})
        self.cur = {"text": [], "link_chars": 0}

    def handle_starttag(self, tag, attrs):
        if tag in DROP_TAGS:
            self.skip += 1
        elif tag == "a":
            self.in_a += 1
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in DROP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag == "a":
            self.in_a = max(0, self.in_a - 1)
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self.skip:
            return
        self.cur["text"].append(data)
        if self.in_a:
            self.cur["link_chars"] += len(data)


def extract_main_text(html: str, min_words: int = 8, max_link_density: float = 0.3) -> str:
    p = _Blocks()
    p.feed(html)
    p._flush()
    # >>> SOLUTION: keep blocks with >= min_words words and link_chars/len(text) <= max_link_density; join with newlines
    keep = [b["text"] for b in p.blocks
            if len(b["text"].split()) >= min_words and b["link_chars"] / max(1, len(b["text"])) <= max_link_density]
    return "\n".join(keep)
    # <<< SOLUTION


def naive_text(html: str) -> str:
    """What a WET-style dump gives you: every text node, chrome included."""
    return " ".join(re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>", " ", html, flags=re.S)).split())


def word_prf(extracted: str, truth: str) -> tuple[float, float]:
    e, t = collections.Counter(extracted.split()), collections.Counter(truth.split())
    tp = sum((e & t).values())
    return tp / max(1, sum(e.values())), tp / max(1, sum(t.values()))


# ------------------------------------------------------------ language ID


def ngrams(text: str, n_max: int = 3) -> list[str]:
    t = f" {text.lower()} "
    return [t[i : i + n] for n in range(1, n_max + 1) for i in range(len(t) - n + 1)]


class LangID:
    """Multinomial naive Bayes over character 1-3 grams with add-one smoothing."""

    def fit(self, samples: dict[str, list[str]]):
        self.langs = sorted(samples)
        self.counts = {l: collections.Counter(g for s in samples[l] for g in ngrams(s)) for l in self.langs}
        self.totals = {l: sum(c.values()) for l, c in self.counts.items()}
        self.vocab = len(set().union(*self.counts.values()))
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        # >>> SOLUTION: log P(text|lang) = sum over n-grams of log((count+1)/(total+V)); softmax over languages (uniform prior)
        logp = {}
        grams = ngrams(text)
        for l in self.langs:
            c, tot = self.counts[l], self.totals[l] + self.vocab
            logp[l] = sum(math.log((c.get(g, 0) + 1) / tot) for g in grams)
        m = max(logp.values())
        z = sum(math.exp(v - m) for v in logp.values())
        return {l: math.exp(v - m) / z for l, v in logp.items()}
        # <<< SOLUTION

    def predict(self, text: str) -> tuple[str, float]:
        p = self.predict_proba(text)
        best = max(p, key=p.get)
        return best, p[best]


def train_langid(seed: int = 0) -> LangID:
    rnd = random.Random(seed)
    return LangID().fit({l: [good_doc(rnd, l) for _ in range(40)] for l in ("en", "de", "fr", "es")})


def langid_accuracy_by_length(model: LangID, seed: int = 1) -> list[dict]:
    rnd = random.Random(seed)
    rows = []
    for n_words in (1, 2, 30):
        ok = conf = 0
        trials = 200
        for _ in range(trials):
            l = rnd.choice(["en", "de", "fr", "es"])
            text = sentence(rnd, l, (n_words, n_words))
            pred, p = model.predict(text)
            ok += pred == l
            conf += p
        rows.append({"words": n_words, "accuracy": ok / trials, "mean confidence": conf / trials})
    return rows


def extraction_report(n: int = 50, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    rows = {"ours": [0.0, 0.0], "naive (WET-like)": [0.0, 0.0]}
    for _ in range(n):
        truth = good_doc(rnd)
        html = to_html(rnd, truth, title="Report")
        for name, fn in (("ours", extract_main_text), ("naive (WET-like)", naive_text)):
            p, r = word_prf(fn(html), truth)
            rows[name][0] += p / n
            rows[name][1] += r / n
    return [{"extractor": k, "word precision": v[0], "word recall": v[1]} for k, v in rows.items()]


def _demo():
    print(table(extraction_report(), ["extractor", "word precision", "word recall"]))
    print(table(langid_accuracy_by_length(train_langid()), ["words", "accuracy", "mean confidence"]))


def report() -> str:
    """The gate measurement as text (used by tools/gates.py)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _demo()
    return buf.getvalue()


if __name__ == "__main__":
    print(report())
