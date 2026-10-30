"""Week 14: an LSM-tree key-value store.

Roadmap: "An LSM tree: write-ahead log, memtable, sorted SSTables, Bloom
filters, compaction. Measure write, read and space amplification."

Why it matters here: the crawler's URL-seen table, the dedup index and the
link library are all write-heavy key-value workloads - the case LSM trees
(RocksDB, LevelDB, Cassandra) are built for.

  put/delete  append to the WAL (durability), then update the memtable
  flush       when the memtable is big, write it as an immutable sorted SSTable
              with a sparse index and a Bloom filter, then truncate the WAL
  get         memtable, then SSTables newest -> oldest; the Bloom filter lets a
              lookup skip tables that cannot contain the key
  compaction  merge SSTables so reads probe fewer files and dead versions are
              dropped. Two policies, two trade-offs:
                tiered  - merge once N similar-sized tables pile up (cheap writes)
                leveled - keep ONE sorted run; merge every flush into it (cheap
                          reads and space, expensive writes)
Amplification (what you report):
  write amp = bytes written to disk / bytes of user data written
  read amp  = SSTables actually read per get (after Bloom filtering)
  space amp = bytes on disk / bytes of live data
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from common.bench import table

TOMB = None  # tombstone value


class Bloom:
    def __init__(self, n: int, bits_per_key: int = 10):
        self.m = max(64, n * bits_per_key)
        self.k = 7
        self.bits = bytearray((self.m + 7) // 8)

    def _pos(self, key: str):
        d = hashlib.blake2b(key.encode(), digest_size=16).digest()
        a, b = int.from_bytes(d[:8], "big"), int.from_bytes(d[8:], "big") | 1
        return ((a + i * b) % self.m for i in range(self.k))

    def add(self, key):
        for p in self._pos(key):
            self.bits[p >> 3] |= 1 << (p & 7)

    def __contains__(self, key):
        return all(self.bits[p >> 3] >> (p & 7) & 1 for p in self._pos(key))


class SSTable:
    """Sorted JSON-lines file + in-memory sparse index and Bloom filter."""

    INDEX_EVERY = 16

    def __init__(self, path: str):
        self.path = path
        self.tier = 0
        self.index: list[tuple[str, int]] = []
        keys = []
        with open(path, "rb") as f:
            off = 0
            for i, line in enumerate(f):
                k = json.loads(line)[0]
                keys.append(k)
                if i % self.INDEX_EVERY == 0:
                    self.index.append((k, off))
                off += len(line)
        self.bloom = Bloom(len(keys))
        for k in keys:
            self.bloom.add(k)
        self.size = os.path.getsize(path)

    @staticmethod
    def write(path: str, items: list[tuple[str, object]]) -> int:
        """items must be sorted by key. Returns bytes written."""
        data = b"".join(json.dumps([k, v]).encode() + b"\n" for k, v in items)
        with open(path, "wb") as f:
            f.write(data)
        return len(data)

    def get(self, key: str):
        """(found, value). Uses the sparse index to read only one block."""
        # >>> SOLUTION: binary-search the sparse index for the last entry with key <= target; seek there; scan at most INDEX_EVERY lines
        lo, hi = 0, len(self.index) - 1
        if not self.index or key < self.index[0][0]:
            return False, None
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.index[mid][0] <= key:
                lo = mid
            else:
                hi = mid - 1
        with open(self.path, "rb") as f:
            f.seek(self.index[lo][1])
            for _ in range(self.INDEX_EVERY):
                line = f.readline()
                if not line:
                    break
                k, v = json.loads(line)
                if k == key:
                    return True, v
                if k > key:
                    break
        return False, None
        # <<< SOLUTION

    def items(self):
        with open(self.path) as f:
            for line in f:
                k, v = json.loads(line)
                yield k, v


class LSM:
    def __init__(self, directory: str, memtable_limit: int = 200, policy: str = "tiered", tier_size: int = 4):
        self.dir = pathlib.Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.limit, self.policy, self.tier_size = memtable_limit, policy, tier_size
        self.mem: dict[str, object] = {}
        self.tables: list[SSTable] = []  # oldest first
        self.seq = 0
        self.stats = {"user_bytes": 0, "disk_bytes": 0, "gets": 0, "tables_read": 0}
        for p in sorted(self.dir.glob("sst-*.jsonl")):
            self.tables.append(SSTable(str(p)))
            self.seq = max(self.seq, int(p.stem.split("-")[1]) + 1)
        self.wal_path = self.dir / "wal.log"
        self._replay()
        self.wal = open(self.wal_path, "ab")

    # ---------------- durability
    def _replay(self):
        if not self.wal_path.exists():
            return
        # >>> SOLUTION: read each JSON line [key, value] of the WAL into the memtable (a torn last line is ignored)
        with open(self.wal_path, "rb") as f:
            for line in f:
                try:
                    k, v = json.loads(line)
                except ValueError:
                    break
                self.mem[k] = v
        # <<< SOLUTION

    def _log(self, key, value):
        rec = json.dumps([key, value]).encode() + b"\n"
        self.wal.write(rec)
        self.wal.flush()
        self.stats["disk_bytes"] += len(rec)

    # ---------------- API
    def put(self, key: str, value: str):
        self.stats["user_bytes"] += len(key) + len(value)
        self._log(key, value)
        self.mem[key] = value
        if len(self.mem) >= self.limit:
            self.flush()

    def delete(self, key: str):
        self._log(key, TOMB)
        self.mem[key] = TOMB
        if len(self.mem) >= self.limit:
            self.flush()

    def get(self, key: str):
        self.stats["gets"] += 1
        if key in self.mem:
            return self.mem[key]
        # >>> SOLUTION: newest table first; skip tables whose Bloom filter excludes the key; count each table actually read in stats['tables_read']; first hit wins (a tombstone means deleted -> None)
        for t in reversed(self.tables):
            if key not in t.bloom:
                continue
            self.stats["tables_read"] += 1
            found, v = t.get(key)
            if found:
                return v
        return None
        # <<< SOLUTION

    def flush(self):
        if not self.mem:
            return
        path = str(self.dir / f"sst-{self.seq:06d}.jsonl")
        self.seq += 1
        self.stats["disk_bytes"] += SSTable.write(path, sorted(self.mem.items()))
        self.tables.append(SSTable(path))
        self.mem.clear()
        self.wal.close()
        self.wal = open(self.wal_path, "wb")  # truncate: everything in it is now in an SSTable
        self.compact()

    # ---------------- compaction
    def _merge(self, tables: list[SSTable], drop_tombstones: bool) -> SSTable:
        merged: dict[str, object] = {}
        for t in tables:              # oldest first, so newer values overwrite
            for k, v in t.items():
                merged[k] = v
        items = sorted((k, v) for k, v in merged.items() if not (drop_tombstones and v is TOMB))
        path = str(self.dir / f"sst-{self.seq:06d}.jsonl")
        self.seq += 1
        self.stats["disk_bytes"] += SSTable.write(path, items)
        for t in tables:
            os.remove(t.path)
        return SSTable(path)

    def compact(self):
        # >>> SOLUTION: leveled -> if 2+ tables, merge ALL into one (dropping tombstones, it is the only run); tiered -> once the newest tier_size tables exist, merge them (drop tombstones only if they include the oldest table)
        if self.policy == "leveled":
            if len(self.tables) >= 2:
                self.tables = [self._merge(self.tables, drop_tombstones=True)]
        elif self.policy == "tiered":
            # Tables carry a tier (flush = 0, merge of tier t = t+1). Tiers never
            # increase toward the newest end, so "the newest tier_size tables of one
            # tier" is always a CONTIGUOUS suffix: merging it keeps newest-wins order.
            # (Merging non-adjacent tables would let an old value jump ahead of a newer one.)
            while len(self.tables) >= self.tier_size:
                group = self.tables[-self.tier_size :]
                if len({t.tier for t in group}) != 1:
                    break
                includes_oldest = group[0] is self.tables[0]
                merged = self._merge(group, drop_tombstones=includes_oldest)
                merged.tier = group[0].tier + 1
                self.tables = self.tables[: -self.tier_size] + [merged]
        # <<< SOLUTION

    # ---------------- metrics
    def amplification(self, live: dict) -> dict:
        disk = sum(t.size for t in self.tables)
        live_bytes = sum(len(json.dumps([k, v])) + 1 for k, v in live.items())
        return {"write amp": self.stats["disk_bytes"] / max(1, self.stats["user_bytes"]),
                "read amp": self.stats["tables_read"] / max(1, self.stats["gets"]),
                "space amp": disk / max(1, live_bytes), "tables": len(self.tables)}

    def close(self):
        self.wal.close()


def workload(db: LSM, n_ops: int = 6000, keys: int = 1500, seed: int = 0) -> dict:
    rnd = random.Random(seed)
    model: dict[str, str] = {}
    for _ in range(n_ops):
        k = f"https://site{rnd.randrange(keys) % 50}.example.com/p{rnd.randrange(keys)}"
        if rnd.random() < 0.1 and model:
            dk = rnd.choice(list(model))
            db.delete(dk)
            model.pop(dk, None)
        else:
            v = f"crawled:{rnd.randrange(10**6)}"
            db.put(k, v)
            model[k] = v
    db.flush()
    for k in rnd.sample(list(model), min(500, len(model))):
        assert db.get(k) == model[k]
    for i in range(500):
        db.get(f"https://never.example.com/{i}")
    return model


def report() -> str:
    import tempfile

    rows = []
    for policy in ("tiered", "leveled"):
        with tempfile.TemporaryDirectory() as d:
            db = LSM(d, policy=policy)
            live = workload(db)
            rows.append({"policy": policy, **db.amplification(live)})
            db.close()
    return table(rows, ["policy", "write amp", "read amp", "space amp", "tables"])


if __name__ == "__main__":
    print(report())
