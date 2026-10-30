# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 19: perceptual hashing and near-duplicate image removal.

Roadmap: "aHash/dHash/pHash; robustness to resize, recompression, crop, flip;
Hamming search at scale with multi-index hashing."

Web image pools are full of the same picture re-encoded, resized, watermarked
and cropped. Webster et al. (2023, "On the De-duplication of LAION-2B",
arXiv:2303.12733) estimate "roughly 700 million images, or about 30%" of
LAION-2B are likely duplicated. Exact byte hashes catch none of the re-encodes;
perceptual hashes map an image to 64 bits such that small edits flip few bits.

  aHash  downscale to 8x8 grey, bit = pixel > mean
  dHash  downscale to 9x8 grey, bit = pixel > right neighbour (gradients)
  pHash  downscale to 32x32 grey, 2-D DCT, keep the 8x8 lowest frequencies
         (minus DC), bit = coefficient > median

Search: comparing all pairs is O(n^2). Multi-index hashing (Norouzi et al.
2012): split the 64 bits into m=4 bands of 16; if two hashes differ in at most
r < m bits, by pigeonhole at least one band matches exactly, so exact-match
buckets on each band give every candidate. Verify candidates with popcount.
"""
from __future__ import annotations

import collections
import io
import pathlib
import random
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.fft import dctn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table


def scene(rnd: random.Random, size: int = 96) -> Image.Image:
    """A random 'photo': gradient background + several shapes. Distinct seeds -> distinct images."""
    rng = np.random.default_rng(rnd.randrange(2**32))
    yy, xx = np.mgrid[:size, :size] / size
    base = rng.uniform(0, 255, 3)
    grad = rng.uniform(-120, 120, (2, 3))
    img = base + yy[..., None] * grad[0] + xx[..., None] * grad[1]
    for _ in range(rnd.randint(3, 7)):
        cy, cx, r = rng.uniform(0, size, 2).tolist() + [rng.uniform(size / 12, size / 4)]
        col = rng.uniform(0, 255, 3)
        if rnd.random() < 0.5:
            m = (yy * size - cy) ** 2 + (xx * size - cx) ** 2 < r * r
        else:
            m = (abs(yy * size - cy) < r) & (abs(xx * size - cx) < r * 0.6)
        img[m] = col
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


TRANSFORMS = {
    "resize 50%": lambda im: im.resize((im.width // 2, im.height // 2), Image.BILINEAR),
    "jpeg q=25": lambda im: Image.open(io.BytesIO(_jpeg(im, 25))),
    "brightness +25%": lambda im: ImageEnhance.Brightness(im).enhance(1.25),
    "crop 8%": lambda im: im.crop((int(im.width * .04), int(im.height * .04), int(im.width * .96), int(im.height * .96))),
    "watermark": lambda im: _watermark(im),
    "h-flip": lambda im: ImageOps.mirror(im),
}


def _jpeg(im, q):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return b.getvalue()


def _watermark(im):
    a = np.array(im).copy()
    h, w = a.shape[:2]
    a[int(h * .85):, int(w * .55):] = 255   # white text box in a corner
    return Image.fromarray(a)


def _grey(im: Image.Image, w: int, h: int) -> np.ndarray:
    return np.asarray(im.convert("L").resize((w, h), Image.LANCZOS), dtype=np.float64)


def _bits_to_int(bits: np.ndarray) -> int:
    return int("".join("1" if b else "0" for b in bits.ravel()), 2)


def ahash(im: Image.Image) -> int:
    g = _grey(im, 8, 8)
    return _bits_to_int(g > g.mean())


def dhash(im: Image.Image) -> int:
    # TODO: 9 wide x 8 high grey image; bit = left pixel > right pixel (64 bits)
    raise NotImplementedError("TODO(w19_image_dedup): 9 wide x 8 high grey image; bit = left pixel > right pixel (64 bits)")


def phash(im: Image.Image) -> int:
    # TODO: 32x32 grey, 2-D DCT-II (orthonormal), top-left 8x8 block, median taken without the DC term, bit = coef > median
    raise NotImplementedError("TODO(w19_image_dedup): 32x32 grey, 2-D DCT-II (orthonormal), top-left 8x8 block, median taken without the DC term, bit = coef > median")


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class MultiIndex:
    """Find all stored hashes within radius r (< bands) of a query."""

    def __init__(self, bands: int = 4, bits: int = 64):
        self.m, self.w = bands, bits // bands
        self.tables = [collections.defaultdict(list) for _ in range(bands)]
        self.items: list[int] = []
        self.compared = 0

    def _band(self, h: int, i: int) -> int:
        return (h >> (i * self.w)) & ((1 << self.w) - 1)

    def add(self, h: int) -> int:
        idx = len(self.items)
        self.items.append(h)
        for i in range(self.m):
            self.tables[i][self._band(h, i)].append(idx)
        return idx

    def query(self, h: int, r: int) -> list[int]:
        assert r < self.m, "pigeonhole guarantee needs r < bands"
        # TODO: union of bucket members over all bands; keep those with hamming <= r; count verifications in self.compared
        raise NotImplementedError("TODO(w19_image_dedup): union of bucket members over all bands; keep those with hamming <= r; count verifications in self.compared")


def dedup(hashes: list[int], r: int) -> list[int]:
    """Indices to keep: first occurrence of each near-duplicate cluster (greedy)."""
    mi = MultiIndex(bands=8 if r >= 4 else 4)
    keep = []
    for i, h in enumerate(hashes):
        if not mi.query(h, r):
            keep.append(i)
        mi.add(h)
    return keep


def robustness(n: int = 300, seed: int = 0) -> list[dict]:
    rnd = random.Random(seed)
    originals = [scene(rnd) for _ in range(n)]
    rows = []
    for name, fn in (("aHash", ahash), ("dHash", dhash), ("pHash", phash)):
        H = [fn(im) for im in originals]
        # distances between different images give the false-positive side
        diff = [hamming(H[i], H[j]) for i in range(n) for j in range(i + 1, min(n, i + 6))]
        row = {"hash": name, "distinct p1 dist": float(np.percentile(diff, 1))}
        for t, tf in TRANSFORMS.items():
            row[t] = float(np.median([hamming(H[i], fn(tf(originals[i]))) for i in range(n)]))
        rows.append(row)
    return rows


def report() -> str:
    rows = robustness()
    cols = ["hash", "distinct p1 dist"] + list(TRANSFORMS)
    rnd = random.Random(1)
    base = [scene(rnd) for _ in range(2000)]
    pool, truth = [], []
    for i, im in enumerate(base):
        pool.append(im); truth.append(i)
        if rnd.random() < 0.3:
            t = rnd.choice(["resize 50%", "jpeg q=25", "brightness +25%", "watermark"])
            pool.append(TRANSFORMS[t](im)); truth.append(i)
    planted = len(pool) - len(base)
    dd = []
    for name, fn in (("dHash", dhash), ("pHash", phash)):
        H = [fn(im) for im in pool]
        for r in (3, 7):
            keep = dedup(H, r)
            surv = collections.Counter(truth[i] for i in keep)
            dd.append({"hash": name, "radius": r, "kept": len(keep),
                       "planted dups removed": (planted - sum(v - 1 for v in surv.values())) / planted,
                       "distinct images lost": len(base) - len(surv)})
    mi = MultiIndex()
    for h in H:
        mi.add(h)
    mi.compared = 0
    for h in H[:500]:
        mi.query(h, 3)
    return ("Median Hamming distance (64 bits) after each edit, and the 1st percentile between DIFFERENT images:\n" +
            table(rows, cols) +
            f"\n\nDedup of {len(pool)} images ({planted} planted near-dups: resize/jpeg/brightness/watermark):\n" +
            table(dd, ["hash", "radius", "kept", "planted dups removed", "distinct images lost"]) +
            f"\nmulti-index (4x16-bit bands, r=3): {mi.compared / 500:.1f} candidates verified per query vs {len(H)} brute force")


if __name__ == "__main__":
    print(report())
