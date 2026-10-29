import torch


def test_weight_store_versions(lab):
    m = lab.new_policy()
    store = lab.WeightStore(m)
    other = lab.new_policy(seed=1)
    assert store.pull(other, -1) == 0
    with torch.no_grad():
        next(m.parameters()).add_(1.0)
    store.publish(m)
    assert store.pull(other, 0) == 1
    assert torch.equal(next(other.parameters()), next(m.parameters()))
    assert store.pull(other, 1) == 1  # nothing newer


def test_staleness_bound_respected(lab):
    s = lab.train_async(updates=12, max_staleness=1)
    assert max(s["staleness"]) <= 1


def test_zero_staleness_is_on_policy(lab):
    s = lab.train_async(updates=6, max_staleness=0)
    assert set(s["staleness"]) == {0}


def test_async_learns(lab):
    s = lab.train_async(updates=30, max_staleness=2)
    assert s["acc_end"] > s["acc_start"] + 0.08, s
