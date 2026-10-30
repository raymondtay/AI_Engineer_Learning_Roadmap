"""Week 9: model-based quality filtering and decontamination.

Roadmap: "Train a fastText-style quality classifier and an educational-value
regressor; n-gram overlap decontamination against eval sets."

Quality classifier (the DCLM recipe in miniature): a linear model over hashed
unigram + bigram features (what fastText is), trained with positives that look
like what you want more of (DCLM: OpenHermes 2.5 + ELI5) against random web
text, then keep the top 10% by score. Heuristic filters (Week 6) remove
garbage; a classifier ranks *among* acceptable pages, which heuristics cannot.

Decontamination: if benchmark items appear in training data, scores measure
memorisation. Index every 13-gram of the eval set and drop training documents
that contain one (13 words is the GPT-3 convention; tune it and report).

Synthetic stand-in for "educational": explanatory prose built around a topic
with connectives such as "because" and "for example".
"""
from __future__ import annotations

import pathlib
import random
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np

from common.bench import prf, table
from common.webgen import TOPIC, WORDS, good_doc, perturb, sentence

CONNECTIVES = ["because", "for example", "this means that", "as a result", "in other words", "which explains why"]


def _explanatory(rnd, topic):
    w = [rnd.choice(WORDS["en"]) for _ in range(rnd.randint(6, 12))]
    w[rnd.randrange(len(w))] = topic
    return f"The {topic} {' '.join(w)} {rnd.choice(CONNECTIVES)} {' '.join(rnd.choice(WORDS['en']) for _ in range(5))}."


def edu_doc(rnd: random.Random, p_explain: float = 0.35) -> str:
    """Mostly ordinary prose; about a third of sentences explain a single topic."""
    topic = rnd.choice(TOPIC)
    sents = [_explanatory(rnd, topic) if rnd.random() < p_explain else
             sentence(rnd) for _ in range(rnd.randint(10, 16))]
    return " ".join(sents)


def web_doc(rnd: random.Random, p_explain: float = 0.08) -> str:
    """Random web prose that occasionally explains something too: classes overlap, as in reality."""
    topic = rnd.choice(TOPIC)
    sents = [_explanatory(rnd, topic) if rnd.random() < p_explain else
             sentence(rnd) for _ in range(rnd.randint(10, 16))]
    return " ".join(sents)


# ------------------------------------------------------------------ features + model

def features(text: str, dim: int = 1 << 18) -> dict[int, float]:
    w = text.lower().split()
    # >>> SOLUTION: hash unigrams and bigrams (crc32 % dim) into a sparse count dict; L2-normalise the values
    f: dict[int, float] = {}
    for g in w + [a + " " + b for a, b in zip(w, w[1:])]:
        h = zlib.crc32(g.encode()) % dim
        f[h] = f.get(h, 0.0) + 1.0
    norm = sum(v * v for v in f.values()) ** 0.5 or 1.0
    return {k: v / norm for k, v in f.items()}
    # <<< SOLUTION


class LinearClassifier:
    """Logistic regression on sparse hashed features, trained by SGD (fastText's model is this plus embeddings)."""

    def __init__(self, dim: int = 1 << 18):
        self.w = np.zeros(dim)
        self.b = 0.0
        self.dim = dim

    def score(self, text: str) -> float:
        f = features(text, self.dim)
        z = self.b + sum(self.w[k] * v for k, v in f.items())
        return 1 / (1 + np.exp(-z))

    def fit(self, pos: list[str], neg: list[str], epochs: int = 5, lr: float = 0.5, seed: int = 0):
        data = [(features(t, self.dim), 1.0) for t in pos] + [(features(t, self.dim), 0.0) for t in neg]
        rnd = random.Random(seed)
        for _ in range(epochs):
            rnd.shuffle(data)
            # >>> SOLUTION: for each (f, y): p = sigmoid(b + w.f); g = p - y; w[k] -= lr*g*v for k,v in f; b -= lr*g
            for f, y in data:
                z = self.b + sum(self.w[k] * v for k, v in f.items())
                g = 1 / (1 + np.exp(-z)) - y
                for k, v in f.items():
                    self.w[k] -= lr * g * v
                self.b -= lr * g
            # <<< SOLUTION
        return self


def auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Probability a random positive outscores a random negative."""
    s = np.array(scores_pos)[:, None] - np.array(scores_neg)[None, :]
    return float((s > 0).mean() + 0.5 * (s == 0).mean())


def top_fraction(texts: list[str], model: LinearClassifier, frac: float = 0.1) -> list[int]:
    scores = np.array([model.score(t) for t in texts])
    k = max(1, int(len(texts) * frac))
    return sorted(np.argsort(-scores)[:k].tolist())


# ------------------------------------------------------------------ decontamination

def ngram_set(text: str, n: int = 13) -> set[int]:
    w = text.lower().split()
    return {zlib.crc32(" ".join(w[i : i + n]).encode()) for i in range(len(w) - n + 1)}


def contaminated(train: list[str], eval_items: list[str], n: int = 13) -> list[int]:
    """Indices of training documents sharing any n-gram with any eval item."""
    # >>> SOLUTION: union all eval n-gram hashes into one set; flag a training doc if any of its n-grams is in it
    index = set().union(*(ngram_set(e, n) for e in eval_items))
    return [i for i, t in enumerate(train) if not index.isdisjoint(ngram_set(t, n))]
    # <<< SOLUTION


# ------------------------------------------------------------------ experiment

def experiment(seed: int = 0) -> dict:
    rnd = random.Random(seed)
    pos_train = [edu_doc(rnd) for _ in range(300)]
    neg_train = [web_doc(rnd) for _ in range(300)]
    # 10% label noise: proxy labels (like DCLM's OH-2.5/ELI5 positives) are never clean
    flip = set(rnd.sample(range(300), 30))
    pos_train, neg_train = ([t for i, t in enumerate(pos_train) if i not in flip] + [t for i, t in enumerate(neg_train) if i in flip],
                            [t for i, t in enumerate(neg_train) if i not in flip] + [t for i, t in enumerate(pos_train) if i in flip])
    model = LinearClassifier().fit(pos_train, neg_train)
    test_pos = [edu_doc(rnd) for _ in range(100)]
    test_neg = [web_doc(rnd) for _ in range(900)]
    a = auc([model.score(t) for t in test_pos], [model.score(t) for t in test_neg])
    pool = test_pos + test_neg
    top = set(top_fraction(pool, model, 0.1))
    return {"auc": a, "top10%": prf(top, set(range(100)))}


def decontam_experiment(seed: int = 0, n: int = 13, edit_rate: float = 0.03) -> dict:
    rnd = random.Random(seed)
    eval_items = [" ".join(rnd.choice(WORDS["en"] + TOPIC) for _ in range(30)) + "?" for _ in range(50)]
    train = [good_doc(rnd) for _ in range(400)]
    planted = set(rnd.sample(range(400), 20))
    for i in planted:  # paste an eval question, lightly edited, into the page
        train[i] = train[i] + "\n" + perturb(rnd, rnd.choice(eval_items), edit_rate)
    return prf(set(contaminated(train, eval_items, n)), planted)


def _demo():
    r = experiment()
    print(f"quality classifier AUC {r['auc']:.3f}; top-10% precision {r['top10%']['precision']:.2f} recall {r['top10%']['recall']:.2f}")
    rows = []
    for n in (8, 13, 20):
        for rate in (0.0, 0.05, 0.15):
            m = decontam_experiment(n=n, edit_rate=rate)
            rows.append({"n-gram": n, "edit rate": rate, "precision": m["precision"], "recall": m["recall"]})
    print(table(rows, ["n-gram", "edit rate", "precision", "recall"]))


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
