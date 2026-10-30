import tempfile
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest


def batch(lo, hi, **extra):
    cols = {"doc_id": pa.array(range(lo, hi), pa.int64()), "lang": pa.array(["en"] * (hi - lo))}
    cols.update(extra)
    return pa.table(cols)


def test_append_and_time_travel(lab):
    t = lab.Table(tempfile.mkdtemp())
    s1 = t.append(batch(0, 100))
    t.append(batch(100, 250))
    assert t.scan().num_rows == 250
    assert t.scan(snapshot_id=s1).num_rows == 100


def test_pruning_by_stats(lab):
    t = lab.Table(tempfile.mkdtemp())
    for i in range(8):
        t.append(batch(i * 100, i * 100 + 100))
    r = t.scan(where=("doc_id", 310, 320))
    assert r.num_rows == 11 and t.files_read == 1


def test_schema_evolution(lab):
    t = lab.Table(tempfile.mkdtemp())
    t.append(batch(0, 10))
    t.append(batch(10, 20, quality=pa.array([0.9] * 10)))
    r = t.scan()
    assert r["quality"].null_count == 10
    t.rename_column("lang", "language")
    r = t.scan(columns=["doc_id", "language"])
    assert r["language"].to_pylist() == ["en"] * 20   # old files read through the rename


def test_commit_is_atomic_cas(lab):
    t = lab.Table(tempfile.mkdtemp())
    t.append(batch(0, 1))
    base = t.current_version()
    meta = t.metadata()
    t._commit(base, meta)
    with pytest.raises(lab.CommitConflict):
        t._commit(base, meta)


def test_concurrent_writers_lose_nothing(lab):
    t = lab.Table(tempfile.mkdtemp())
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda i: t.append(batch(i * 10, i * 10 + 10)), range(24)))
    ids = sorted(t.scan()["doc_id"].to_pylist())
    assert ids == list(range(240))


def test_compaction_keeps_history(lab):
    t = lab.Table(tempfile.mkdtemp())
    s = [t.append(batch(i * 10, i * 10 + 10)) for i in range(5)]
    t.compact()
    assert len(t._snapshot_files(t.metadata())) == 1
    assert t.scan().num_rows == 50
    assert t.scan(snapshot_id=s[2]).num_rows == 30
