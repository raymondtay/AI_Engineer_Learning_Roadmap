import collections
import random
import subprocess
import sys
import tempfile

import pyarrow.parquet as pq


def test_stable_hash_is_stable_across_processes(lab):
    here = lab.stable_hash("site7.example.com")
    code = ("import sys,importlib.util as u;s=u.spec_from_file_location('m',sys.argv[1]);m=u.module_from_spec(s);"
            "sys.modules['m']=m;s.loader.exec_module(m);print(m.stable_hash('site7.example.com'))")
    out = subprocess.run([sys.executable, "-c", code, lab.__file__], capture_output=True, text=True,
                         env={"PYTHONHASHSEED": "123", "PATH": ""}, cwd=str(__import__("pathlib").Path(lab.__file__).parents[3]))
    assert int(out.stdout.strip()) == here


def test_host_stats_matches_single_process(lab):
    d = tempfile.mkdtemp()
    shards = lab.make_shards(4, 800, d)
    truth = collections.defaultdict(lambda: [0, 0])
    for s in shards:
        t = pq.read_table(s)
        for h, txt in zip(t["host"].to_pylist(), t["text"].to_pylist()):
            truth[h][0] += 1
            truth[h][1] += len(txt.split())
    got = lab.host_stats(shards, workers=2, n_parts=5)
    assert got == {k: tuple(v) for k, v in truth.items()}


def test_external_sort(lab):
    rnd = random.Random(0)
    xs = [rnd.randrange(10**6) for _ in range(5000)]
    out = list(lab.external_sort(iter(xs), chunk=700, spill_dir=tempfile.mkdtemp()))
    assert out == sorted(xs)
