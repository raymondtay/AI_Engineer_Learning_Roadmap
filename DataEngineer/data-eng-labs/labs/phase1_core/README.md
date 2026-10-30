# Phase 1 — Core data engineering (Weeks 1–4)

The habits every later phase depends on: columnar storage, reading query
plans, counting in tokens, and parallel jobs that give the same answer every run.

Reference numbers: 2-core CPU VM, `LAB=solution python tools/gates.py` (see `GATES_solution.md`).

## Week 1 — Columnar formats (`w01_columnar`)

`count(lang = 'de')` over 20,000 synthetic web docs:

| layout | file MB | MB scanned | ms |
| --- | --- | --- | --- |
| JSONL | 26.7 | 26.7 | 200 |
| Parquet, Snappy | 12.3 | 0.0056 | 6.6 |
| Parquet, Zstd | 9.4 | 0.0043 | 5.3 |
| Parquet, sorted by `lang` | 8.6 | 0.00009 | 1.5 |

JSONL must be read in full, always. Parquet reads one column (projection), and
sorting by the filter column makes row-group min/max statistics exclusive enough
that almost every row group is skipped. That is the whole argument for
"sort/cluster by what you filter on".

## Week 2 — Query engines (`w02_query_engines`)

Three corpus questions in pandas (given), DuckDB, PyArrow compute and Spark (local);
tests check all engines agree, Spark included (pyspark 4.2.1).

| engine | Q1 ms | Q2 ms | Q3 ms |
| --- | --- | --- | --- |
| pandas | 26.9 | 19.5 | 32.5 |
| DuckDB | 13.7 | 8.2 | 9.8 |
| PyArrow | 10.3 | 13.5 | — |

DuckDB's `EXPLAIN` shows `Filters: lang = 'de'` inside `READ_PARQUET` — the
filter is pushed into the scan, which is what makes Week 1's numbers happen.
Spark is not timed here: on 20k rows its JVM start-up dominates and the number
would be meaningless. (PySpark 4.2.1 warns that it does not fully support
pandas 3; the agreement test passes regardless.)

## Week 3 — Tokens (`w03_tokens`)

Byte-level BPE (vocab 800) trained mostly on English:

| eval set | tokens / byte |
| --- | --- |
| English prose | 0.23 |
| German prose | 0.39 |
| French prose | 0.42 |
| spam | 0.58 |
| code | 0.88 |

The same bytes cost 1.7–1.9x more tokens in German or French than English with
this tokenizer — a mixture decision hiding inside the tokenizer. Budgets and
mixture weights in later weeks are therefore set in tokens, not bytes.

## Week 4 — Parallel map/shuffle/reduce (`w04_parallel`)

Per-host statistics over 8 shards with a stable-hash shuffle (Python's `hash()`
of `str` is randomised per process — partitioning with it silently sends the
same key to different reducers) and an external sort.

| workers | seconds | speed-up |
| --- | --- | --- |
| 1 | 0.77 | 1.00 |
| 2 | 0.44 | 1.74 |
| 4 | 0.45 | 1.71 |

Flat beyond the 2 physical cores, as it should be. A first gates run measured
0.94x because a test suite was running concurrently — scaling must be measured
on an idle machine.

## Sources

* Apache Parquet file format (row groups, column chunks, statistics) — https://parquet.apache.org/docs/file-format/
* DuckDB, "EXPLAIN" and Parquet filter pushdown — https://duckdb.org/docs/
* Sennrich, Haddow, Birch, "Neural Machine Translation of Rare Words with Subword Units", ACL 2016 (BPE); Radford et al., GPT-2 report, 2019 (byte-level BPE).
* Python docs, `PYTHONHASHSEED` (hash randomisation of str/bytes).
* Amdahl, "Validity of the single processor approach…", AFIPS 1967.
