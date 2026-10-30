# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 16: an Iceberg-shaped table format on plain files.

Roadmap: "A lake table: immutable data files, manifests with column stats,
snapshots, an atomic metadata pointer. Time travel, schema evolution,
concurrent writers, compaction."

The design follows the Apache Iceberg spec (iceberg.apache.org/spec), shrunk:

  data/*.parquet              immutable data files, never rewritten in place
  metadata/v{N}.json          table metadata: schema (columns have stable ids),
                              list of snapshots, current snapshot id
  metadata/snap-{id}.json     the snapshot's manifest: every live data file with
                              row count and per-column min/max (for pruning)
  metadata/version-hint.txt   pointer to the current v{N}.json

A commit writes a new v{N+1}.json and then *atomically claims* that name
(os.link fails if it exists - the moral equivalent of a catalog's
compare-and-swap). Two writers racing for v{N+1}: one wins, the other re-reads
the new base and retries (optimistic concurrency). Readers never see half a
commit because they only follow fully-written metadata files.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import sys
import time
import uuid

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table


class CommitConflict(Exception):
    pass


class Table:
    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "metadata").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- metadata io
    def _hint(self) -> pathlib.Path:
        return self.root / "metadata" / "version-hint.txt"

    def current_version(self) -> int:
        h = self._hint()
        v = int(h.read_text()) if h.exists() else 0
        while (self.root / "metadata" / f"v{v + 1}.json").exists():   # hint may lag: walk forward
            v += 1
        return v

    def metadata(self, version: int | None = None) -> dict:
        v = self.current_version() if version is None else version
        if v == 0:
            return {"schema": [], "next_col_id": 1, "snapshots": [], "current": None}
        return json.loads((self.root / "metadata" / f"v{v}.json").read_text())

    def _commit(self, base_version: int, meta: dict):
        """Atomically make `meta` version base_version+1, or raise CommitConflict."""
        tmp = self.root / "metadata" / f".tmp-{uuid.uuid4().hex}.json"
        tmp.write_text(json.dumps(meta))
        target = self.root / "metadata" / f"v{base_version + 1}.json"
        # TODO: os.link(tmp, target) succeeds only if target does not exist (atomic create); on FileExistsError raise CommitConflict; always remove tmp; on success update version-hint.txt
        raise NotImplementedError("TODO(w16_lake_table): os.link(tmp, target) succeeds only if target does not exist (atomic create); on FileExistsError raise CommitConflict; always remove tmp; on success update version-hint.txt")

    def _write_hint(self, v: int):
        # The hint is only an optimisation (readers walk forward from it), but a reader
        # must never see it half-written: write a temp file, then rename over it.
        tmp = self.root / "metadata" / f".hint-{uuid.uuid4().hex}"
        tmp.write_text(str(v))
        os.replace(tmp, self._hint())

    def _snapshot_files(self, meta: dict, snapshot_id: int | None = None) -> list[dict]:
        sid = meta["current"] if snapshot_id is None else snapshot_id
        if sid is None:
            return []
        return json.loads((self.root / "metadata" / f"snap-{sid}.json").read_text())

    # ---------------------------------------------------------- writes
    def _write_file(self, tbl: pa.Table, meta: dict) -> dict:
        name = f"data/{uuid.uuid4().hex}.parquet"
        # store columns under their stable ids, so a rename never touches data files
        ids = {c["name"]: c["id"] for c in meta["schema"]}
        out = tbl.rename_columns([str(ids[n]) for n in tbl.column_names])
        pq.write_table(out, self.root / name)
        stats = {}
        for n in tbl.column_names:
            col = tbl[n]
            if pa.types.is_integer(col.type) or pa.types.is_floating(col.type) or pa.types.is_string(col.type):
                mm = pc.min_max(col)
                stats[str(ids[n])] = [mm["min"].as_py(), mm["max"].as_py()]
        return {"path": name, "rows": tbl.num_rows, "stats": stats}

    def _evolve(self, meta: dict, tbl: pa.Table) -> dict:
        meta = json.loads(json.dumps(meta))
        known = {c["name"] for c in meta["schema"]}
        for f in tbl.schema:
            if f.name not in known:   # schema evolution: add column with a fresh id
                meta["schema"].append({"id": meta["next_col_id"], "name": f.name, "type": str(f.type)})
                meta["next_col_id"] += 1
        return meta

    def _commit_files(self, op: str, build, retries: int = 30):
        """build(meta, live_files) -> (meta', new_live_files). Retried on conflict with
        jittered exponential backoff: without it, 8 writers retrying in lock-step can
        keep colliding (livelock) - it happened in this lab's first version.
        A losing attempt may leave an unreferenced data file behind; Iceberg cleans
        those up with its "remove orphan files" maintenance action."""
        for attempt in range(retries):
            if attempt:
                time.sleep(random.uniform(0, min(0.2, 0.002 * 2 ** attempt)))
            base = self.current_version()
            meta = self.metadata(base)
            live = self._snapshot_files(meta)
            meta2, files = build(meta, live)
            # Racing writers share the same base, so ids derived from it would collide and
            # the loser would overwrite the winner's manifest. Random ids cannot collide.
            sid = uuid.uuid4().int >> 66
            (self.root / "metadata" / f"snap-{sid}.json").write_text(json.dumps(files))
            meta2["snapshots"].append({"id": sid, "op": op, "parent": meta2["current"], "version": base + 1})
            meta2["current"] = sid
            try:
                self._commit(base, meta2)
                return sid
            except CommitConflict:
                continue
        raise CommitConflict("gave up")

    def append(self, tbl: pa.Table) -> int:
        def build(meta, live):
            meta = self._evolve(meta, tbl)
            return meta, live + [self._write_file(tbl, meta)]
        return self._commit_files("append", build)

    def rename_column(self, old: str, new: str) -> int:
        def build(meta, live):
            meta = json.loads(json.dumps(meta))
            for c in meta["schema"]:
                if c["name"] == old:
                    c["name"] = new
            return meta, live
        return self._commit_files("rename", build)

    def compact(self, target_rows: int = 10**9) -> int:
        """Rewrite all live files into one. Old files stay for time travel."""
        def build(meta, live):
            t = self._read_files(meta, live, None, None)
            return meta, [self._write_file(t, meta)] if t.num_rows else []
        return self._commit_files("replace", build)

    # ---------------------------------------------------------- reads
    def _read_files(self, meta: dict, files: list[dict], columns, where) -> pa.Table:
        by_id = {str(c["id"]): c for c in meta["schema"]}
        names = columns or [c["name"] for c in meta["schema"]]
        name_to_id = {c["name"]: str(c["id"]) for c in meta["schema"]}
        parts = []
        for f in files:
            # TODO: prune: where=(col, lo, hi) skips the file if its [min,max] for col does not overlap [lo,hi]; then read the file, map id-named columns back to current names, fill columns the file predates with nulls, and filter rows
            raise NotImplementedError("TODO(w16_lake_table): prune: where=(col, lo, hi) skips the file if its [min,max] for col does not overlap [lo,hi]; then read the file, map id-named columns back to current names, fill columns the file predates with nulls, and filter rows")
        self.files_read = len(parts)
        if not parts:
            return pa.table({n: pa.array([], pa.type_for_alias(by_id[name_to_id[n]]["type"])) for n in names})
        return pa.concat_tables(parts)

    def scan(self, columns=None, where=None, snapshot_id: int | None = None) -> pa.Table:
        meta = self.metadata()
        return self._read_files(meta, self._snapshot_files(meta, snapshot_id), columns, where)

    def snapshots(self) -> list[dict]:
        return self.metadata()["snapshots"]


def report() -> str:
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    rows = []
    with tempfile.TemporaryDirectory() as d:
        t = Table(d)
        for day in range(10):   # one file per crawl day, sorted ids -> tight stats
            t.append(pa.table({"doc_id": pa.array(range(day * 1000, day * 1000 + 1000), pa.int64()),
                               "score": pa.array([0.5] * 1000)}))
        full = t.scan()
        t.scan(where=("doc_id", 4200, 4300))
        rows.append({"what": "pruned scan doc_id in [4200,4300]", "value": f"{t.files_read}/10 files read"})
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda i: t.append(pa.table({"doc_id": pa.array([10**6 + i], pa.int64()),
                                                      "score": pa.array([1.0])})), range(16)))
        rows.append({"what": "16 concurrent appends", "value": f"{t.scan().num_rows - full.num_rows} rows landed"})
        before = len(t.snapshots())
        t.compact()
        rows.append({"what": "after compaction", "value": f"{len(t._snapshot_files(t.metadata()))} live file(s), {before} older snapshots still readable"})
    return table(rows, ["what", "value"])


if __name__ == "__main__":
    print(report())
