# Phase 4 — Data infrastructure (Weeks 14–17)

The JD's "data infrastructure" direction: the storage and execution layers a
pre-training pipeline runs on. You build small, correct versions of the four
systems you will otherwise only configure — then measure the trade-offs that
decide which one to configure how.

Reference numbers: 2-core CPU VM, `python <lab>/solution.py`.

## Week 14 — LSM tree (`w14_lsm`)

WAL + memtable + sorted SSTables with a sparse index and a Bloom filter (~1% FP
at 10 bits/key, 7 hashes), crash recovery by WAL replay, tombstones, and two
compaction policies. Correctness is checked against a Python `dict` under 3,000
random puts/deletes per policy.

| policy | write amp | read amp (tables read / get) | space amp | tables |
| --- | --- | --- | --- | --- |
| tiered (merge 4 same-tier runs) | 4.33 | 0.53 | 1.13 | 6 |
| leveled (one sorted run) | 19.05 | 0.50 | 1.00 | 1 |

The textbook trade-off, measured: leveled pays ~4x the write I/O to keep one run
(lowest space and read cost); tiered writes cheaply and keeps more runs.
"Read amp" is below 1 because half the gets are for absent keys that every
Bloom filter rejects.

**Bug worth knowing about (it happened while building this lab):** a first
tiered policy merged "similar-sized" tables that were not adjacent in age. A
merged run then sat *newer* than a table it should be older than, and a deleted
key came back. Merges must take a contiguous age range; the fix tracks a tier
per table so the newest N same-tier tables are always contiguous.

## Week 15 — Log with exactly-once (`w15_log_queue`)

Partitioned append-only log (crc32 key partitioning), idempotent producer
(per-partition sequence numbers, the broker drops repeats, survives restarts),
and a consumer whose sink is SQLite. 2,000 sends, 5% of acks lost, 5% chance of
a crash at each batch commit:

| mode | idempotent producer | log records | restarts | lost | duplicates |
| --- | --- | --- | --- | --- | --- |
| at-most-once (commit offset, then write) | yes | 2000 | 19 | 133 | 0 |
| at-least-once (write, then commit offset) | yes | 2000 | 19 | 0 | 133 |
| exactly-once (one transaction) | **no** | 2137 | 20 | 0 | 137 |
| exactly-once (one transaction) | yes | 2000 | 19 | 0 | **0** |

Row 3 is the lesson: a transactional consumer cannot remove duplicates the
producer already wrote. Kafka needs both halves (KIP-98: idempotent producer +
transactions).

## Week 16 — Lake table (`w16_lake_table`)

Iceberg-shaped: immutable Parquet data files; per-snapshot manifests with
row counts and per-column min/max; table metadata `v{N}.json`; commits claim
`v{N+1}.json` atomically (`os.link` fails if it exists — the stand-in for a
catalog compare-and-swap) and retry on conflict. Columns are stored under
stable ids, so renames never rewrite data and old files read through them.

| check | result |
| --- | --- |
| scan `doc_id in [4200,4300]` over 10 day-files | 1/10 files read (min/max pruning) |
| 16 concurrent appends (8 threads) | 16/16 rows landed |
| compaction | 1 live file; all 26 earlier snapshots still readable (time travel) |

Two concurrency bugs found and fixed while building it, both instructive:
the version hint was written non-atomically (a reader saw an empty file), and
snapshot ids derived from the base version collided between racing writers so
the loser overwrote the winner's manifest. Random ids + write-temp-then-rename fix both.

## Week 17 — DAG engine + Rust hot path (`w17_dag_engine`)

Topological scheduling (Kahn), parallel independent tasks, retries, failure
isolation (downstream of a failure is skipped, other branches finish), and a
cache keyed by `sha256(name, version, params, upstream keys)`.

| run | tasks run | from cache | seconds |
| --- | --- | --- | --- |
| cold | 5 | 0 | 1.44 |
| warm, nothing changed | 0 | 5 | 0.006 |
| one filter parameter changed | 3 | 2 | 1.95 |

Hot path: MinHash (k=64, word 5-gram shingles) in Python vs a std-only Rust
`cdylib` called through `ctypes` (no pyo3, so it builds offline), checked bit
for bit against the Python reference:

| impl | ms/doc (run 1) | ms/doc (run 2, `GATES_solution.md`) |
| --- | --- | --- |
| Python | 3.36 | 2.95 |
| Rust | 0.041 | 0.028 |
| speed-up | **82x** | **104x** |

Quote it as "roughly 80–100x on this box"; single-run timings on a shared 2-core VM move by 20%+.

The Rust file is given, not an exercise; your TODOs are the engine and the
ctypes binding. Tests needing cargo are marked `rust` and skip without it.

## Caveats

* Single machine. Week 15's crash model is "exception at a random point", not
  a partitioned network; Week 16's CAS is a local file-system link, which is
  atomic on POSIX file systems but **not** on S3 — that is why real Iceberg
  deployments commit through a catalog (REST, Glue, Nessie, Polaris).
* The LSM's numbers depend on the workload (6,000 ops, 10% deletes, 1,500 hot
  keys). Change the update ratio and re-measure before quoting them.

## Sources

* O'Neil et al., "The Log-Structured Merge-Tree", Acta Informatica 1996; RocksDB wiki, "Compaction" (leveled vs universal/tiered).
* Bloom, "Space/Time Trade-offs in Hash Coding with Allowable Errors", CACM 1970.
* Apache Kafka KIP-98, "Exactly Once Delivery and Transactional Messaging".
* Apache Iceberg table spec — https://iceberg.apache.org/spec/
* Kahn, "Topological sorting of large networks", CACM 1962.
