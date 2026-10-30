import random


def imgs(lab, n=60, seed=0):
    rnd = random.Random(seed)
    return [lab.scene(rnd) for _ in range(n)]


def test_hashes_are_64_bit_and_deterministic(lab):
    im = imgs(lab, 1)[0]
    for fn in (lab.ahash, lab.dhash, lab.phash):
        h = fn(im)
        assert 0 <= h < 2**64 and fn(im) == h


def test_dhash_definition(lab):
    import numpy as np
    from PIL import Image
    ramp = Image.fromarray(np.tile(np.arange(0, 255, 255 / 90).astype(np.uint8)[:90][::-1], (80, 1)))
    assert lab.dhash(ramp) == 2**64 - 1      # left brighter than right everywhere
    assert lab.dhash(ramp.transpose(Image.FLIP_LEFT_RIGHT)) == 0


def test_robust_to_mild_edits_not_to_flip(lab):
    for im in imgs(lab, 30):
        h = lab.phash(im)
        assert lab.hamming(h, lab.phash(lab.TRANSFORMS["resize 50%"](im))) <= 4
        assert lab.hamming(h, lab.phash(lab.TRANSFORMS["jpeg q=25"](im))) <= 6
    flips = [lab.hamming(lab.phash(im), lab.phash(lab.TRANSFORMS["h-flip"](im))) for im in imgs(lab, 30, 1)]
    assert sorted(flips)[15] > 16


def test_multi_index_equals_brute_force(lab):
    rnd = random.Random(0)
    base = [rnd.getrandbits(64) for _ in range(300)]
    hs = base + [b ^ (1 << rnd.randrange(64)) ^ (1 << rnd.randrange(64)) for b in base[:100]]
    mi = lab.MultiIndex()
    for h in hs:
        mi.add(h)
    for q in hs[:150]:
        assert mi.query(q, 3) == [j for j, h in enumerate(hs) if lab.hamming(h, q) <= 3]
    assert mi.compared < 150 * 10


def test_dedup_removes_reencodes(lab):
    base = imgs(lab, 80, 2)
    pool = base + [lab.TRANSFORMS["jpeg q=25"](im) for im in base[:40]]
    keep = lab.dedup([lab.phash(im) for im in pool], r=3)
    assert set(range(80)) <= set(keep) | set()  # originals come first and survive
    assert len(keep) <= 80 + 4
