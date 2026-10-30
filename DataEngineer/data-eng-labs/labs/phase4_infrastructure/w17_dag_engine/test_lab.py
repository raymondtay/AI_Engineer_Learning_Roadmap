import tempfile
import threading
import time

import pytest


def test_topo_and_cycles(lab):
    dag = lab.DAG(tempfile.mkdtemp())
    for n, deps in [("c", ["a", "b"]), ("a", []), ("b", ["a"]), ("d", ["c"])]:
        dag.add(lab.Task(n, lambda *x: 0, deps=deps))
    order = dag.topo()
    assert order.index("a") < order.index("b") < order.index("c") < order.index("d")
    dag.add(lab.Task("a", lambda *x: 0, deps=["d"]))
    with pytest.raises(ValueError):
        dag.topo()


def test_results_flow(lab):
    dag = lab.DAG(tempfile.mkdtemp())
    dag.add(lab.Task("x", lambda v: v, params={"v": 3}))
    dag.add(lab.Task("y", lambda x: x * 10, deps=["x"]))
    dag.add(lab.Task("z", lambda x, y: x + y, deps=["x", "y"]))
    assert dag.run()["z"] == 33


def test_cache_and_invalidation(lab):
    d = tempfile.mkdtemp()
    dag = lab.pipeline(d, n_docs=60)
    dag.run()
    assert len(dag.events("ran")) == 5
    dag.run()
    assert dag.events("ran") == [] and len(dag.events("cached")) == 5
    dag = lab.pipeline(d, min_words=60, n_docs=60)
    dag.run()
    assert dag.events("ran") == ["filter", "minhash", "report"]   # load and stats reused


def test_retry_and_failure_isolation(lab):
    dag = lab.DAG(tempfile.mkdtemp())
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise IOError("transient")
        return 1

    dag.add(lab.Task("flaky", flaky, retries=3))
    dag.add(lab.Task("bad", lambda: 1 / 0))
    dag.add(lab.Task("after_bad", lambda x: x, deps=["bad"]))
    dag.add(lab.Task("after_after", lambda x: x, deps=["after_bad"]))
    dag.add(lab.Task("ok", lambda f: f + 1, deps=["flaky"]))
    res = dag.run()
    assert res["ok"] == 2
    assert dag.state["bad"] == "failed"
    assert dag.events("skipped") == ["after_after", "after_bad"]
    assert dag.events("retry") == ["flaky", "flaky"]


def test_independent_tasks_run_in_parallel(lab):
    dag = lab.DAG(tempfile.mkdtemp(), workers=4)
    live, peak = [0], [0]
    lock = threading.Lock()

    def work(i):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.2)
        with lock:
            live[0] -= 1
        return i

    for i in range(4):
        dag.add(lab.Task(f"t{i}", work, params={"i": i}))
    dag.run()
    assert peak[0] >= 3


@pytest.mark.rust
def test_rust_matches_python(lab):
    from common.webgen import corpus

    docs = [x["text"] for x in corpus(40, seed=3)] + ["one two", ""]
    rs = lab.minhash_rust(docs)
    assert rs.tolist() == [lab.minhash_py(d) for d in docs]
