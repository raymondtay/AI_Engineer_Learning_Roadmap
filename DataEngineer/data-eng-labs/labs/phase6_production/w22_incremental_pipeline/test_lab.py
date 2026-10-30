import tempfile

import pytest


@pytest.fixture(scope="module")
def bs():
    from conftest import load_lab
    return load_lab(__file__).batches(n_batches=4, per=150)


def test_incremental_equals_full(lab, bs):
    p = lab.Pipeline(tempfile.mkdtemp())
    for i, b in enumerate(bs):
        p.run_batch(f"b{i}", b)
    full = lab.Pipeline(tempfile.mkdtemp())
    full.run_batch("all", [x for b in bs for x in b])
    assert p.kept_ids() == full.kept_ids()
    assert p.processed == sum(len(b) for b in bs)


def test_rerun_is_noop(lab, bs):
    d = tempfile.mkdtemp()
    p = lab.Pipeline(d)
    p.run_batch("b0", bs[0])
    before = p.kept_ids()
    assert lab.Pipeline(d).run_batch("b0", bs[0])["skipped"]
    assert lab.Pipeline(d).kept_ids() == before


def test_crash_before_commit_then_retry(lab, bs):
    d = tempfile.mkdtemp()
    p = lab.Pipeline(d)
    p.run_batch("b0", bs[0])
    with pytest.raises(lab.Crash):
        p.run_batch("b1", bs[1], crash_after_output=True)
    assert p.state["manifest"] == ["b0"]                  # in-memory state did not advance
    q = lab.Pipeline(d)                                   # "restarted process"
    assert q.state["manifest"] == ["b0"]
    q.run_batch("b1", bs[1])
    ref = lab.Pipeline(tempfile.mkdtemp())
    ref.run_batch("b0", bs[0]); ref.run_batch("b1", bs[1])
    assert q.kept_ids() == ref.kept_ids()


def test_cross_batch_duplicates_removed(lab, bs):
    p = lab.Pipeline(tempfile.mkdtemp())
    p.run_batch("b0", bs[0])
    r = p.run_batch("b0-again-under-new-id", bs[0])
    assert r["kept"] == 0
