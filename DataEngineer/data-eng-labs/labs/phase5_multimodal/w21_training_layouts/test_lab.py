import tempfile

import pytest


@pytest.fixture(scope="module")
def shards():
    from conftest import load_lab
    lab = load_lab(__file__)
    d = tempfile.mkdtemp()
    return lab.write_tar(lab.samples(1000), d, per_shard=100)


def test_tar_roundtrip(lab):
    ss = lab.samples(250)
    paths = lab.write_tar(ss, tempfile.mkdtemp(), per_shard=100)
    assert len(paths) == 3
    out = [s for p in paths for s in lab.iter_tar(p)]
    assert out == ss


def test_parquet_and_files_roundtrip(lab):
    ss = lab.samples(50)
    d = tempfile.mkdtemp()
    lab.write_parquet(ss, d + "/x.parquet")
    assert lab.read_parquet(d + "/x.parquet") == ss
    lab.write_files(ss, d + "/f")
    assert lab.read_files(d + "/f") == ss


def test_stream_covers_each_epoch_once(lab, shards):
    st = lab.ShuffledStream(shards, seed=1, buffer=64)
    first = [next(st)["key"] for _ in range(1000)]
    assert len(set(first)) > 930   # buffer straddles the epoch boundary, so a few repeat early
    assert first != sorted(first)


def test_resume_is_exact(lab, shards):
    ref = lab.ShuffledStream(shards, seed=3, buffer=50)
    full = [next(ref)["key"] for _ in range(700)]
    a = lab.ShuffledStream(shards, seed=3, buffer=50)
    part = [next(a)["key"] for _ in range(333)]
    b = lab.ShuffledStream(shards, seed=3, buffer=50)   # a fresh process
    b.load_state_dict(a.state_dict())
    rest = [next(b)["key"] for _ in range(367)]
    assert part + rest == full


def test_mixture(lab):
    m = lab.Mixture({"a": 100, "b": 100}, {"a": 3, "b": 1}, seed=0)
    r = m.run(20000)
    assert abs(r["a"]["share"] - 0.75) < 0.02
    assert m.source_at(1234) == lab.Mixture({"a": 1, "b": 1}, {"a": 3, "b": 1}, seed=0).source_at(1234)
    sizes, w = {"web": 10**6, "books": 1000}, {"web": 0.9, "books": 0.1}
    cw = lab.Mixture.capped_weights(sizes, w, steps=100_000, max_epochs=4)
    assert cw["books"] == pytest.approx(0.04) and sum(cw.values()) == pytest.approx(1)
