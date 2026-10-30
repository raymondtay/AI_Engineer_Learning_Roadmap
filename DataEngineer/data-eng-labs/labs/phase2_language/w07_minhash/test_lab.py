import random

import numpy as np
import pytest

from common.webgen import good_doc, perturb


def test_s_curve_matches_fineweb_paper(lab):
    # FineWeb (Penedo et al. 2024): 14 buckets x 8 hashes -> 56%, 77%, 92%, 98.8%
    for s, expect in [(0.70, 0.56), (0.75, 0.77), (0.80, 0.92), (0.85, 0.988)]:
        assert lab.s_curve(s, 14, 8) == pytest.approx(expect, abs=0.006)


def test_signature_estimates_jaccard(lab):
    rnd = random.Random(0)
    a = good_doc(rnd)
    b = perturb(rnd, a, 0.05)
    sa, sb = lab.shingles(a), lab.shingles(b)
    mh = lab.MinHasher(512)
    est = float(np.mean(mh.signature(sa) == mh.signature(sb)))
    assert est == pytest.approx(lab.jaccard(sa, sb), abs=0.05)


def test_lsh_precision_recall(lab):
    rows = {r["bands x rows"]: r for r in lab.evaluate(lab.make_dataset(60))}
    fw = rows["14x8"]
    assert fw["recall"] > 0.85 and fw["precision"] > 0.6, fw


def test_dedup_keeps_one_per_cluster(lab):
    rnd = random.Random(2)
    src = [good_doc(rnd) for _ in range(20)]
    texts = src + [perturb(rnd, s, 0.01) for s in src]
    keep = lab.dedup(texts)
    assert all(i in keep for i in range(20))          # one representative per source
    # LSH is probabilistic: a pair at Jaccard 0.86 escapes ~1% of the time (S-curve).
    # Only pairs well above the threshold are guaranteed in practice.
    for i in keep:
        if i >= 20:
            assert lab.jaccard(lab.shingles(texts[i]), lab.shingles(texts[i - 20])) < 0.9
