import math

import pytest

from common import dist as ldist


@pytest.mark.parametrize("world", [2, 4])
def test_tp_matches_single_device(lab, world):
    out = ldist.run(lab.tp_check, world)
    for r in range(world):
        out_diff, dx_diff = out[r]
        assert out_diff < 1e-5 and dx_diff < 1e-4


def test_bubble(lab):
    assert math.isclose(lab.bubble_fraction(4, 4), 3 / 7)
    assert lab.bubble_fraction(4, 64) < 0.05


def test_schedule_shape(lab):
    s = lab.schedule_1f1b(4, 6, 0)
    assert s[:3] == [("F", 0), ("F", 1), ("F", 2)]  # warm-up = p - 1 - stage
    assert [x for x in s if x[0] == "F"] == [("F", i) for i in range(6)]
    assert [x for x in s if x[0] == "B"] == [("B", i) for i in range(6)]
    last = lab.schedule_1f1b(4, 6, 3)
    assert last[:2] == [("F", 0), ("B", 0)]  # last stage has no warm-up
    # Never more than p - stage micro-batches in flight (the 1F1B memory bound).
    inflight = peak = 0
    for op, _ in s:
        inflight += 1 if op == "F" else -1
        peak = max(peak, inflight)
    assert peak == 4


@pytest.mark.parametrize("world,micro", [(2, 4), (4, 6)])
def test_pipeline_grads_match(lab, world, micro):
    ref = lab.reference_grads(world, micro)
    out = ldist.run(lab.pipeline_train_step, world, micro)
    for r in range(world):
        for a, b in zip(out[r], ref[r]):
            assert (a - b).abs().max().item() < 1e-6
