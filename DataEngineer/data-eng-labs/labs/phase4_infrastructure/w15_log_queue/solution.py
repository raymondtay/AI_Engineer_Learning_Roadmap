"""Week 15: a partitioned log with exactly-once processing.

Roadmap: "A Kafka-shaped log: partitions, offsets, consumer groups; idempotent
producer; exactly-once via an atomic offset+output commit. Kill it mid-run."

The crawl -> clean -> dedup stages of a real pipeline are joined by queues. When
a worker dies mid-batch, three things can go wrong:
  * the producer retried a send whose ack was lost   -> duplicate record in the log
  * the consumer processed records but died before committing its offset
                                                     -> records processed twice
  * the consumer committed its offset but died before writing output
                                                     -> records lost
Kafka's answer (KIP-98) is an idempotent producer (sequence numbers per
producer, broker drops repeats) plus transactions that commit consumer offsets
together with output. Here the sink is SQLite, so "together" is one SQL
transaction: output rows and the new offset commit or roll back as a unit.

You will measure, under injected crashes:  at-most-once loses, at-least-once
duplicates, exactly-once does neither.
"""
from __future__ import annotations

import json
import pathlib
import random
import sqlite3
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table


class Crash(Exception):
    """Injected failure: the process 'dies' here."""


# ------------------------------------------------------------------ broker

class Log:
    """One directory, one append-only file per partition; offset = record index."""

    def __init__(self, directory: str, partitions: int = 4):
        self.dir = pathlib.Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = partitions
        self.records: list[list[dict]] = [[] for _ in range(partitions)]
        self.last_seq: dict[tuple[str, int], int] = {}   # (producer, partition) -> last seq accepted
        for p in range(partitions):
            f = self.dir / f"p{p}.log"
            if f.exists():
                for line in f.read_text().splitlines():
                    r = json.loads(line)
                    self.records[p].append(r)
                    self.last_seq[(r["producer"], p)] = r["seq"]

    def partition_for(self, key: str) -> int:
        return zlib.crc32(key.encode()) % self.n   # stable across processes (unlike hash())

    def append(self, p: int, producer: str, seq: int, key: str, value: str, idempotent: bool) -> int:
        """Returns the record's offset. With idempotence, a repeated (producer, seq) is a no-op."""
        # >>> SOLUTION: if idempotent and seq <= last accepted seq for (producer, p): return the existing offset without appending; else append to memory and the partition file, remember seq
        if idempotent and seq <= self.last_seq.get((producer, p), -1):
            for off, r in enumerate(self.records[p]):
                if r["producer"] == producer and r["seq"] == seq:
                    return off
        rec = {"producer": producer, "seq": seq, "key": key, "value": value}
        self.records[p].append(rec)
        with open(self.dir / f"p{p}.log", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.last_seq[(producer, p)] = seq
        return len(self.records[p]) - 1
        # <<< SOLUTION

    def read(self, p: int, offset: int, max_n: int) -> list[dict]:
        return self.records[p][offset : offset + max_n]


class Producer:
    def __init__(self, log: Log, name: str, idempotent: bool, lost_ack_rate: float = 0.0, seed: int = 0):
        self.log, self.name, self.idem = log, name, idempotent
        self.seq: dict[int, int] = {}
        self.rnd = random.Random(seed)
        self.lost = lost_ack_rate

    def send(self, key: str, value: str) -> int:
        p = self.log.partition_for(key)
        s = self.seq.get(p, 0)
        while True:
            off = self.log.append(p, self.name, s, key, value, self.idem)
            if self.rnd.random() >= self.lost:   # ack arrived
                break
            # ack lost on the network: the producer cannot know the write landed, so it retries
        self.seq[p] = s + 1
        return off


# ------------------------------------------------------------------ consumers

def _db(path: str) -> sqlite3.Connection:
    c = sqlite3.connect(path, isolation_level=None)
    c.execute("CREATE TABLE IF NOT EXISTS out (key TEXT, value TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS offsets (part INTEGER PRIMARY KEY, next INTEGER)")
    return c


def consume(log: Log, db_path: str, mode: str, batch: int = 7, crash_rate: float = 0.0, seed: int = 0) -> int:
    """Process every partition to the end; restart after each injected Crash.
    mode: at_most_once | at_least_once | exactly_once. Returns restarts."""
    rnd = random.Random(seed)
    restarts = 0
    while True:
        db = _db(db_path)
        try:
            for p in range(log.n):
                row = db.execute("SELECT next FROM offsets WHERE part=?", (p,)).fetchone()
                off = row[0] if row else 0
                while True:
                    recs = log.read(p, off, batch)
                    if not recs:
                        break
                    new = off + len(recs)
                    rows = [(r["key"], r["value"].upper()) for r in recs]   # the "processing"
                    _commit(db, mode, p, new, rows, rnd, crash_rate)
                    off = new
            db.close()
            return restarts
        except Crash:
            db.close()     # an open transaction is rolled back, as when a process dies
            restarts += 1


def _commit(db, mode, p, new_offset, rows, rnd, crash_rate):
    def maybe_crash():
        if rnd.random() < crash_rate:
            raise Crash()

    def save_offset():
        db.execute("INSERT OR REPLACE INTO offsets VALUES (?, ?)", (p, new_offset))

    def write_rows():
        db.executemany("INSERT INTO out VALUES (?, ?)", rows)

    # >>> SOLUTION: at_most_once: save offset, maybe crash, write rows. at_least_once: write rows, maybe crash, save offset. exactly_once: BEGIN; write rows; maybe crash; save offset; COMMIT (a crash leaves the transaction uncommitted)
    if mode == "at_most_once":
        save_offset(); maybe_crash(); write_rows()
    elif mode == "at_least_once":
        write_rows(); maybe_crash(); save_offset()
    elif mode == "exactly_once":
        db.execute("BEGIN")
        write_rows(); maybe_crash(); save_offset()
        db.execute("COMMIT")
    else:
        raise ValueError(mode)
    # <<< SOLUTION


# ------------------------------------------------------------------ experiment

def run(mode: str, idempotent: bool, directory: str, n: int = 2000, lost_ack: float = 0.05, crash: float = 0.05) -> dict:
    log = Log(f"{directory}/log")
    prod = Producer(log, "crawler-1", idempotent, lost_ack_rate=lost_ack)
    for i in range(n):
        prod.send(f"https://site{i % 37}.example.com/p{i}", f"doc{i}")
    restarts = consume(log, f"{directory}/sink.db", mode, crash_rate=crash)
    db = _db(f"{directory}/sink.db")
    keys = [k for (k,) in db.execute("SELECT key FROM out")]
    db.close()
    distinct = len(set(keys))
    return {"mode": mode, "idempotent producer": idempotent,
            "log records": sum(len(r) for r in log.records), "restarts": restarts,
            "lost": n - distinct, "duplicates": len(keys) - distinct}


def report() -> str:
    import tempfile

    rows = []
    for mode, idem in [("at_most_once", True), ("at_least_once", True), ("exactly_once", False), ("exactly_once", True)]:
        with tempfile.TemporaryDirectory() as d:
            rows.append(run(mode, idem, d))
    return ("2000 sends, 5% lost acks, 5% crash chance per batch commit:\n" +
            table(rows, ["mode", "idempotent producer", "log records", "restarts", "lost", "duplicates"]))


if __name__ == "__main__":
    print(report())
