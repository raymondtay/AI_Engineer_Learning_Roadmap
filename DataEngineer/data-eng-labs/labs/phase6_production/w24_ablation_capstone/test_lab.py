import pytest


@pytest.fixture(scope="module")
def setup():
    from conftest import load_lab
    lab = load_lab(__file__)
    pool = lab.raw_pool(1200)
    return lab.variants(pool), lab.encode(lab.reference_docs(40, 999))


def test_variants_are_nested(lab, setup):
    V, _ = setup
    names = ["raw", "filtered", "filtered+dedup", "filtered+dedup+classifier"]
    assert list(V) == names
    for a, b in zip(names, names[1:]):
        assert set(V[b]) <= set(V[a]) and len(V[b]) < len(V[a])


def test_quality_classifier_prefers_reference_text(lab):
    import random
    from common.webgen import spam_doc
    rnd = random.Random(0)
    clf = lab.NgramQuality().fit(lab.reference_docs(100, 1), [spam_doc(rnd) for _ in range(100)])
    good = [clf.score(t) for t in lab.reference_docs(30, 2)]
    bad = [clf.score(spam_doc(rnd)) for _ in range(30)]
    assert min(good) > max(bad)


def test_training_learns_and_filtering_helps(lab, setup):
    V, val = setup
    raw = lab.train_eval(V["raw"], val, steps=120, seed=0)
    filt = lab.train_eval(V["filtered"], val, steps=120, seed=0)
    assert raw < 4.0            # an untrained model is ~7 bits/char (uniform over 128)
    assert filt < raw           # same budget, cleaner data, lower held-out loss
