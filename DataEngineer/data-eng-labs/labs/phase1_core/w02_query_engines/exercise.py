# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 2: query engines - pandas, DuckDB, PyArrow compute, Spark.

Roadmap: "The same aggregations in pandas, DuckDB and Spark (local). Read the
query plans; find where each spills or serialises."

Three questions a data team asks of every new dump:
  Q1 documents and characters per language
  Q2 top 10 hosts by number of 'good' documents
  Q3 mean and p95 document length per kind

pandas is the reference (given). You write DuckDB SQL, PyArrow compute, and a
Spark version. Then you read DuckDB's EXPLAIN to confirm the filter is pushed
into the Parquet scan, which is what made Week 1's numbers possible.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common.bench import table
from common.data import docs_table


# ------------------------------------------------------------- pandas (reference)

def pandas_q1(path):
    df = pd.read_parquet(path, columns=["lang", "n_chars"])
    out = df.groupby("lang").agg(docs=("n_chars", "size"), chars=("n_chars", "sum")).reset_index()
    return sorted(map(tuple, out[["lang", "docs", "chars"]].values.tolist()))


def pandas_q2(path):
    df = pd.read_parquet(path, columns=["host", "kind"])
    s = df[df.kind == "good"].groupby("host").size().sort_values(ascending=False, kind="stable")
    return sorted(((h, int(n)) for h, n in s.items()), key=lambda x: (-x[1], x[0]))[:10]


def pandas_q3(path):
    df = pd.read_parquet(path, columns=["kind", "n_chars"])
    g = df.groupby("kind")["n_chars"]
    return {k: (round(g.mean()[k], 3), round(float(g.quantile(0.95, interpolation="linear")[k]), 3)) for k in g.groups}


# ------------------------------------------------------------- DuckDB

def duck_q1(con, path):
    # TODO: SELECT lang, count(*), sum(n_chars) FROM read_parquet(path) GROUP BY lang; return sorted tuples
    raise NotImplementedError("TODO(w02_query_engines): SELECT lang, count(*), sum(n_chars) FROM read_parquet(path) GROUP BY lang; return sorted tuples")


def duck_q2(con, path):
    # TODO: filter kind='good', group by host, order by count desc then host, limit 10
    raise NotImplementedError("TODO(w02_query_engines): filter kind='good', group by host, order by count desc then host, limit 10")


def duck_q3(con, path):
    # TODO: avg(n_chars) and quantile_cont(n_chars, 0.95) per kind (quantile_cont matches pandas' linear interpolation)
    raise NotImplementedError("TODO(w02_query_engines): avg(n_chars) and quantile_cont(n_chars, 0.95) per kind (quantile_cont matches pandas' linear interpolation)")


def explain_filter(con, path, lang="de") -> str:
    """Physical plan of a filtered count; look for the filter inside the Parquet scan."""
    return "\n".join(r[1] for r in con.execute(
        f"EXPLAIN SELECT count(*) FROM read_parquet('{path}') WHERE lang = '{lang}'").fetchall())


# ------------------------------------------------------------- PyArrow compute

def arrow_q1(path):
    t = pq.read_table(path, columns=["lang", "n_chars"])
    # TODO: t.group_by('lang').aggregate([('n_chars','count'), ('n_chars','sum')])
    raise NotImplementedError("TODO(w02_query_engines): t.group_by('lang').aggregate([('n_chars','count'), ('n_chars','sum')])")


def arrow_q2(path):
    t = pq.read_table(path, columns=["host", "kind"], filters=[("kind", "=", "good")])
    # TODO: group_by host count, then sort by (-count, host) and take 10
    raise NotImplementedError("TODO(w02_query_engines): group_by host count, then sort by (-count, host) and take 10")


# ------------------------------------------------------------- Spark (optional: needs Java + pyspark)

def spark_q1(path):
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.master("local[2]").appName("w02").config("spark.ui.enabled", "false").getOrCreate()
    try:
        # TODO: spark.read.parquet(path).groupBy('lang').agg(count, sum) and collect
        raise NotImplementedError("TODO(w02_query_engines): spark.read.parquet(path).groupBy('lang').agg(count, sum) and collect")
    finally:
        spark.stop()


def report(n: int = 50_000) -> str:
    t = docs_table(n)
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/docs.parquet"
        pq.write_table(t, path, row_group_size=5000)
        con = duckdb.connect()
        rows = []
        for name, fns in {"pandas": (pandas_q1, pandas_q2, pandas_q3),
                          "duckdb": (lambda p: duck_q1(con, p), lambda p: duck_q2(con, p), lambda p: duck_q3(con, p)),
                          "pyarrow": (arrow_q1, arrow_q2, None)}.items():
            row = {"engine": name}
            for i, fn in enumerate(fns, 1):
                if fn is None:
                    continue
                t0 = time.perf_counter()
                fn(path)
                row[f"Q{i} ms"] = (time.perf_counter() - t0) * 1e3
            rows.append(row)
        plan = explain_filter(con, path)
    return table(rows, ["engine", "Q1 ms", "Q2 ms", "Q3 ms"]) + "\n\nDuckDB plan for WHERE lang='de':\n" + plan


if __name__ == "__main__":
    print(report())
