import math

import pytest

from common.simweb import SimWeb


def test_feedback_beats_bfs(lab):
    web = SimWeb(n_hosts=40, pages_per_host=30, n_mirrors=0, seed=0)
    r = {p: lab.crawl(web, p, budget=300)["kept/fetch"] for p in ("bfs", "opic", "yield_ucb")}
    assert r["yield_ucb"] > 1.3 * r["bfs"], r


def test_freshness_formula(lab):
    assert lab.freshness(1.0, 1e-9) == pytest.approx(1.0, abs=1e-6)
    assert lab.freshness(1.0, 1.0) == pytest.approx(1 - math.exp(-1))
    assert lab.freshness(2.0, math.inf) == 0.0
    for lam, I in [(0.5, 1.0), (2.0, 1.0), (0.1, 7.0)]:
        assert lab.simulate_freshness(lam, I) == pytest.approx(lab.freshness(lam, I), abs=0.02)


def test_cho_garcia_molina(lab):
    rates = [0.01, 0.1, 1.0, 5.0] * 50
    f = {p: lab.mean_freshness(rates, lab.allocate(rates, 20.0, p)) for p in ("proportional", "uniform", "optimal")}
    assert f["uniform"] > f["proportional"]          # the counter-intuitive result
    assert f["optimal"] >= f["uniform"] - 1e-9
    opt = lab.allocate(rates, 20.0, "optimal")
    fastest = [x for x, r in zip(opt, rates) if r == 5.0]
    slow = [x for x, r in zip(opt, rates) if r == 0.1]
    assert sum(fastest) < sum(slow)                  # the optimum gives up on pages that change too fast
    assert sum(opt) == pytest.approx(20.0)
