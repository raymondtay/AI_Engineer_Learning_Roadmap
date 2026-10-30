import tempfile


def test_idempotent_producer_dedups_retries(lab):
    d = tempfile.mkdtemp()
    log = lab.Log(d)
    p = lab.Producer(log, "a", idempotent=True, lost_ack_rate=0.3)
    for i in range(300):
        p.send(f"k{i}", "v")
    assert sum(len(r) for r in log.records) == 300
    d2 = tempfile.mkdtemp()
    log2 = lab.Log(d2)
    p2 = lab.Producer(log2, "a", idempotent=False, lost_ack_rate=0.3)
    for i in range(300):
        p2.send(f"k{i}", "v")
    assert sum(len(r) for r in log2.records) > 330


def test_log_survives_restart(lab):
    d = tempfile.mkdtemp()
    log = lab.Log(d)
    p = lab.Producer(log, "a", idempotent=True)
    for i in range(50):
        p.send(f"k{i}", "v")
    log2 = lab.Log(d)
    assert log2.records == log.records
    # the broker remembers sequence numbers across restarts: a stale retry is still dropped
    log2.append(log2.partition_for("k0"), "a", 0, "k0", "v", idempotent=True)
    assert sum(len(r) for r in log2.records) == 50


def test_delivery_semantics(lab):
    r = {}
    for mode in ("at_most_once", "at_least_once", "exactly_once"):
        r[mode] = lab.run(mode, True, tempfile.mkdtemp(), n=1500, crash=0.08)
        assert r[mode]["restarts"] > 0
    assert r["at_most_once"]["lost"] > 0 and r["at_most_once"]["duplicates"] == 0
    assert r["at_least_once"]["lost"] == 0 and r["at_least_once"]["duplicates"] > 0
    assert r["exactly_once"]["lost"] == 0 and r["exactly_once"]["duplicates"] == 0


def test_exactly_once_needs_idempotent_producer(lab):
    r = lab.run("exactly_once", False, tempfile.mkdtemp(), n=1500, lost_ack=0.1)
    assert r["duplicates"] > 0   # the consumer is exact, but the log itself holds duplicates
