"""Week 1: columnar formats - layouts, row-group pruning, bytes scanned.

Roadmap: "Parquet row groups, encodings and compression; Arrow in memory.
Measure scan time and bytes read for row vs columnar, with and without
predicate pushdown."

The question every pipeline asks thousands of times: "give me the rows where
lang = 'de', and only these columns". How much of the file must be read?
  * JSONL: all of it, always.
  * Parquet: only the needed columns (projection) of the row groups whose
    min/max statistics could contain the value (predicate pushdown).
  * Parquet sorted by the filter column: pushdown can skip almost everything.

Run:  python solution.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from common.bench import table
from common.data import docs_table


def write_layouts(t: pa.Table, out: str, row_group_size: int = 2000) -> dict[str, str]:
    """Write the same table five ways; return {layout: path}."""
    out_p = pathlib.Path(out)
    paths = {}
    p = out_p / "docs.jsonl"
    with open(p, "w") as f:
        for row in t.to_pylist():
            f.write(json.dumps(row) + "\n")
    paths["jsonl"] = str(p)
    for codec in ("snappy", "zstd"):
        p = out_p / f"docs_{codec}.parquet"
        pq.write_table(t, p, compression=codec, row_group_size=row_group_size)
        paths[f"parquet_{codec}"] = str(p)
    # >>> SOLUTION: write a zstd Parquet sorted by (lang, host) so row-group min/max stats cluster each language
    p = out_p / "docs_sorted.parquet"
    pq.write_table(t.sort_by([("lang", "ascending"), ("host", "ascending")]), p, compression="zstd",
                   row_group_size=row_group_size)
    paths["parquet_sorted"] = str(p)
    # <<< SOLUTION
    return paths


def row_groups_matching(path: str, column: str, value) -> list[int]:
    """Row groups whose statistics say `column == value` is possible."""
    md = pq.ParquetFile(path).metadata
    col_idx = md.schema.names.index(column)
    keep = []
    # >>> SOLUTION: for each row group read column chunk statistics; keep it unless stats exist and value < min or value > max
    for rg in range(md.num_row_groups):
        st = md.row_group(rg).column(col_idx).statistics
        if st is None or not st.has_min_max or st.min <= value <= st.max:
            keep.append(rg)
    # <<< SOLUTION
    return keep


def bytes_scanned(path: str, columns: list[str], row_groups: list[int]) -> int:
    """Compressed bytes of the requested column chunks in the requested row groups."""
    md = pq.ParquetFile(path).metadata
    idx = [md.schema.names.index(c) for c in columns]
    # >>> SOLUTION: sum total_compressed_size of column(i) for i in idx over the given row groups
    return sum(md.row_group(rg).column(i).total_compressed_size for rg in row_groups for i in idx)
    # <<< SOLUTION


def count_lang(path: str, layout: str, lang: str) -> tuple[int, int, float]:
    """(rows with lang, bytes scanned, seconds) for one layout."""
    t0 = time.perf_counter()
    if layout == "jsonl":
        n = 0
        with open(path) as f:
            for line in f:
                n += json.loads(line)["lang"] == lang
        return n, os.path.getsize(path), time.perf_counter() - t0
    # >>> SOLUTION: prune row groups, read only those with only the lang column, count matches; report bytes for (lang, id)
    rgs = row_groups_matching(path, "lang", lang)
    pf = pq.ParquetFile(path)
    n = 0
    for rg in rgs:
        col = pf.read_row_group(rg, columns=["lang"])["lang"]
        n += pc.sum(pc.equal(col, lang)).as_py() or 0
    return n, bytes_scanned(path, ["lang"], rgs), time.perf_counter() - t0
    # <<< SOLUTION


def dataset_filter_count(path: str, lang: str) -> int:
    """The same query through pyarrow.dataset, which does pushdown for you."""
    return ds.dataset(path, format="parquet").count_rows(filter=pc.field("lang") == lang)


def report(n: int = 20_000, lang: str = "de") -> str:
    t = docs_table(n)
    with tempfile.TemporaryDirectory() as d:
        paths = write_layouts(t, d)
        rows = []
        for layout, p in paths.items():
            cnt, nbytes, sec = count_lang(p, layout, lang)
            rows.append({"layout": layout, "file MB": os.path.getsize(p) / 1e6, "rows": cnt,
                         "MB scanned": nbytes / 1e6, "ms": sec * 1e3})
    return f"query: count(lang = '{lang}') over {n} docs\n" + table(rows, ["layout", "file MB", "rows", "MB scanned", "ms"])


if __name__ == "__main__":
    print(report())
