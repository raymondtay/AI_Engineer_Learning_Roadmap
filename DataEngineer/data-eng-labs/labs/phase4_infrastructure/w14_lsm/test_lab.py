import random
import tempfile

import pytest


@pytest.mark.parametrize("policy", ["tiered", "leveled"])
def test_matches_dict_model(lab, policy):
    d = tempfile.mkdtemp()
    db = lab.LSM(d, memtable_limit=50, policy=policy)
    rnd = random.Random(0)
    model = {}
    for i in range(3000):
        k = f"k{rnd.randrange(400)}"
        if rnd.random() < 0.15:
            db.delete(k)
            model.pop(k, None)
        else:
            db.put(k, str(i))
            model[k] = str(i)
    for i in range(400):
        assert db.get(f"k{i}") == model.get(f"k{i}")


def test_crash_recovery_from_wal(lab):
    d = tempfile.mkdtemp()
    db = lab.LSM(d, memtable_limit=1000)
    for i in range(100):
        db.put(f"k{i}", f"v{i}")
    db.delete("k5")
    # simulate a crash: no flush, no close (the WAL was flushed on every write)
    db2 = lab.LSM(d, memtable_limit=1000)
    assert db2.get("k7") == "v7" and db2.get("k5") is None and db2.get("k99") == "v99"


def test_reopen_after_flush(lab):
    d = tempfile.mkdtemp()
    db = lab.LSM(d, memtable_limit=10)
    for i in range(55):
        db.put(f"k{i:03d}", f"v{i}")
    db.close()
    db2 = lab.LSM(d, memtable_limit=10)
    assert all(db2.get(f"k{i:03d}") == f"v{i}" for i in range(55))


def test_bloom_skips_tables(lab):
    d = tempfile.mkdtemp()
    db = lab.LSM(d, memtable_limit=20, policy="tiered", tier_size=100)  # no compaction: many tables
    for i in range(400):
        db.put(f"k{i}", "x")
    db.flush()
    before = db.stats["tables_read"]
    for i in range(200):
        assert db.get(f"absent{i}") is None
    assert (db.stats["tables_read"] - before) / 200 < 0.3   # ~1% FP per table x 20 tables


def test_amplification_tradeoff(lab):
    rows = {}
    for policy in ("tiered", "leveled"):
        d = tempfile.mkdtemp()
        db = lab.LSM(d, policy=policy)
        live = lab.workload(db)
        rows[policy] = db.amplification(live)
    assert rows["leveled"]["write amp"] > rows["tiered"]["write amp"]
    assert rows["leveled"]["space amp"] <= rows["tiered"]["space amp"]
    assert rows["leveled"]["read amp"] <= rows["tiered"]["read amp"]
