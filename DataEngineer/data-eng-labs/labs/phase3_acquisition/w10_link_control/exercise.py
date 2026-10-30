# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 10: link scale control - URL normalisation, Bloom filters, mirror detection.

Roadmap: "URL normalisation and tracking-parameter stripping; Bloom-filter URL
dedup; mirror and near-duplicate site detection with SimHash. Measure
link-library shrinkage and unique content lost."

  normalise_url  one canonical spelling per resource: lowercase scheme/host,
                 drop default ports and fragments, resolve dot segments, drop
                 tracking/session parameters, sort the rest
  BloomFilter    "seen this URL?" for billions of URLs in fixed memory, with a
                 false-positive rate you choose: m = -n ln p / (ln 2)^2 bits,
                 k = (m / n) ln 2 hashes. A false positive means a URL is never
                 crawled; there are no false negatives.
  SimHash        a 64-bit fingerprint where similar texts differ in few bits;
                 hosts whose pages mostly match another host's pages are mirrors
"""
from __future__ import annotations

import hashlib
import math
import pathlib
import posixpath
import random
import sys
import zlib
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import prf, table
from common.simweb import SimWeb

DROP_PARAMS = {"fbclid", "gclid", "sessionid", "sid", "phpsessid", "ref"}
DEFAULT_PORTS = {"http": 80, "https": 443}


def normalise_url(url: str) -> str:
    sp = urlsplit(url.strip())
    scheme = sp.scheme.lower()
    # TODO: lowercase host; drop default port; resolve dot segments (keep trailing slash); drop utm_* and DROP_PARAMS; sort the remaining query; drop fragment
    raise NotImplementedError("TODO(w10_link_control): lowercase host; drop default port; resolve dot segments (keep trailing slash); drop utm_* and DROP_PARAMS; sort the remaining query; drop fragment")


class BloomFilter:
    def __init__(self, n: int, p: float):
        # TODO: m = ceil(-n ln p / (ln 2)^2); k = max(1, round(m/n * ln 2)); bit array of m bits
        raise NotImplementedError("TODO(w10_link_control): m = ceil(-n ln p / (ln 2)^2); k = max(1, round(m/n * ln 2)); bit array of m bits")

    def _positions(self, key: str):
        d = hashlib.blake2b(key.encode(), digest_size=16).digest()
        h1, h2 = int.from_bytes(d[:8], "big"), int.from_bytes(d[8:], "big") | 1
        return ((h1 + i * h2) % self.m for i in range(self.k))  # Kirsch-Mitzenmacher double hashing

    def add(self, key: str):
        for b in self._positions(key):
            self.bits[b >> 3] |= 1 << (b & 7)

    def __contains__(self, key: str) -> bool:
        return all(self.bits[b >> 3] >> (b & 7) & 1 for b in self._positions(key))


def simhash(text: str, bits: int = 64, n: int = 3) -> int:
    w = text.lower().split()
    v = [0] * bits
    # TODO: for each word n-gram take a 64-bit hash; add +1/-1 per bit; fingerprint bit = 1 where the sum > 0
    raise NotImplementedError("TODO(w10_link_control): for each word n-gram take a 64-bit hash; add +1/-1 per bit; fingerprint bit = 1 where the sum > 0")


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def detect_mirrors(pages_by_host: dict[str, list[str]], max_dist: int = 3, min_share: float = 0.8) -> set[frozenset]:
    """Host pairs where >= min_share of one host's pages have a near-identical page on the other."""
    fps = {h: [simhash(t) for t in texts] for h, texts in pages_by_host.items()}
    hosts = sorted(fps)
    pairs = set()
    # TODO: for each host pair, share of host A's fingerprints within max_dist of some fingerprint of host B; flag if either direction >= min_share
    raise NotImplementedError("TODO(w10_link_control): for each host pair, share of host A's fingerprints within max_dist of some fingerprint of host B; flag if either direction >= min_share")
    return pairs


def report(seed: int = 0) -> dict:
    web = SimWeb(seed=seed)
    rnd = random.Random(seed)
    raw = [web.variant(u, rnd) for u in web.pages for _ in range(3)]
    canon = {normalise_url(u) for u in raw}
    lost = set(web.pages) - canon
    bf = BloomFilter(len(canon), 0.01)
    for u in canon:
        bf.add(u)
    probe = [f"https://unseen{i}.example.org/x" for i in range(20000)]
    fp = sum(u in bf for u in probe) / len(probe)
    by_host: dict[str, list[str]] = {}
    for u, pg in web.pages.items():
        by_host.setdefault(urlsplit(u).hostname, []).append(pg.text)
    found = detect_mirrors({h: v[:8] for h, v in by_host.items()})
    truth = {frozenset((h.name, h.mirror_of)) for h in web.hosts.values() if h.mirror_of}
    return {"raw urls": len(raw), "canonical": len(canon), "shrink": 1 - len(canon) / len(raw),
            "true pages lost": len(lost), "bloom bits/url": bf.m / len(canon), "bloom fp": fp,
            "mirrors": prf(found, truth)}


if __name__ == "__main__":
    for k, v in report().items():
        print(f"{k:16s} {v}")
