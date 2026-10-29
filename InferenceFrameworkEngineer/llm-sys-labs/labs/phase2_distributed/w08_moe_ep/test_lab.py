import math

import pytest
import torch

from common import dist as ldist


def test_router(lab):
    r = lab.Router(16, 8, 2)
    idx, w, probs = r(torch.randn(10, 16))
    assert idx.shape == (10, 2) and torch.allclose(w.sum(-1), torch.ones(10))
    assert torch.allclose(probs.sum(-1), torch.ones(10))


def test_aux_loss_uniform_is_one(lab):
    E, N = 4, 8
    probs = torch.full((N, E), 1 / E)
    idx = torch.arange(N).remainder(E)[:, None]  # perfectly even, k=1
    assert math.isclose(lab.switch_aux_loss(probs, idx, E).item(), 1.0, rel_tol=1e-6)
    skew = torch.zeros(N, 1, dtype=torch.long)
    p_skew = torch.zeros(N, E); p_skew[:, 0] = 1
    assert lab.switch_aux_loss(p_skew, skew, E).item() == pytest.approx(E)


def test_aux_free_balancing_reduces_imbalance(lab):
    before, after = lab.balance_demo()
    assert before > 2.0 and after < 1.3


def test_local_moe_matches_dense_loop(lab):
    router, experts = lab.build()
    x = torch.randn(10, 16)
    idx, w, _ = router(x)
    ref = torch.stack([sum(w[n, j] * experts[idx[n, j]](x[n]) for j in range(idx.shape[1])) for n in range(10)])
    torch.testing.assert_close(lab.moe_forward_local(x, router, experts), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("world", [2, 4])
def test_expert_parallel(lab, world):
    out = ldist.run(lab.ep_check, world)
    assert max(out.values()) < 1e-5
