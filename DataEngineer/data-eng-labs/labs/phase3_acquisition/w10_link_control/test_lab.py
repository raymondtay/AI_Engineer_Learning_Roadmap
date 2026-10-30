import random

import pytest


@pytest.mark.parametrize("messy,clean", [
    ("HTTPS://Example.COM:443/a/./b/../c?utm_source=x&b=2&a=1#frag", "https://example.com/a/c?a=1&b=2"),
    ("http://example.com:80", "http://example.com/"),
    ("http://example.com:8080/x/", "http://example.com:8080/x/"),
    ("https://example.com/p?fbclid=abc&sessionid=9", "https://example.com/p"),
    ("https://example.com/a%7Eb", "https://example.com/a~b"),
])
def test_normalise(lab, messy, clean):
    assert lab.normalise_url(messy) == clean


def test_bloom_sizing_and_fp_rate(lab):
    bf = lab.BloomFilter(10_000, 0.01)
    assert abs(bf.m / 10_000 - 9.585) < 0.01 and bf.k == 7   # 9.59 bits/item, 7 hashes for 1%
    for i in range(10_000):
        bf.add(f"u{i}")
    assert all(f"u{i}" in bf for i in range(10_000))          # no false negatives
    fp = sum(f"x{i}" in bf for i in range(50_000)) / 50_000
    assert 0.005 < fp < 0.02


def test_simhash_similarity(lab):
    from common.webgen import good_doc, perturb

    rnd = random.Random(0)
    a = good_doc(rnd)
    assert lab.hamming(lab.simhash(a), lab.simhash(perturb(rnd, a, 0.01))) <= 6
    assert lab.hamming(lab.simhash(a), lab.simhash(good_doc(rnd))) > 15


def test_report(lab):
    r = lab.report()
    assert r["shrink"] > 0.6 and r["true pages lost"] == 0
    assert r["mirrors"]["recall"] == 1.0 and r["mirrors"]["precision"] == 1.0
