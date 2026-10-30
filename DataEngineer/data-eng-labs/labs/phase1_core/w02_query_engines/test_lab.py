import tempfile

import duckdb
import pyarrow.parquet as pq
import pytest

from common.data import docs_table


@pytest.fixture(scope="module")
def path():
    d = tempfile.mkdtemp()
    p = f"{d}/docs.parquet"
    pq.write_table(docs_table(5000), p, row_group_size=1000)
    return p


def test_duckdb_matches_pandas(lab, path):
    con = duckdb.connect()
    assert lab.duck_q1(con, path) == lab.pandas_q1(path)
    assert lab.duck_q2(con, path) == lab.pandas_q2(path)
    d3, p3 = lab.duck_q3(con, path), lab.pandas_q3(path)
    assert d3.keys() == p3.keys()
    for k in d3:
        assert d3[k] == pytest.approx(p3[k], rel=1e-6)


def test_arrow_matches_pandas(lab, path):
    assert lab.arrow_q1(path) == lab.pandas_q1(path)
    assert lab.arrow_q2(path) == lab.pandas_q2(path)


def test_filter_is_pushed_into_scan(lab, path):
    plan = lab.explain_filter(duckdb.connect(), path)
    # DuckDB prints the pushed-down predicate inside the scan operator.
    assert "PARQUET_SCAN" in plan.upper() or "TABLE_SCAN" in plan.upper() or "READ_PARQUET" in plan.upper()
    assert "lang" in plan and "de" in plan


@pytest.mark.spark
def test_spark_matches_pandas(lab, path):
    assert lab.spark_q1(path) == lab.pandas_q1(path)
