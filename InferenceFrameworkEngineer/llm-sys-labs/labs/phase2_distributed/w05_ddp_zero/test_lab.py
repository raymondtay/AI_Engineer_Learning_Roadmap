import math

import pytest
import torch

from common import dist as ldist


def _max_diff(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def test_bus_bandwidth(lab):
    # 1 GB in 1 s on 8 ranks: algbw 1e9, busbw 1e9 * 14/8
    assert math.isclose(lab.bus_bandwidth(10**9, 1.0, 8), 1.75e9)


def test_ddp_matches_full_batch(lab):
    ref = lab.reference_train("sgd")
    out = ldist.run(lab.ddp_train, 2)
    assert _max_diff(ref, out[0]) < 1e-5
    assert _max_diff(out[0], out[1]) == 0.0  # replicas must stay bit-identical


def test_zero_matches_adam_and_shards_state(lab):
    ref = lab.reference_train("adam")
    out = ldist.run(lab.zero_train, 2)
    params, state = out[0]
    assert _max_diff(ref, params) < 1e-5
    n = sum(p.numel() for p in lab.make_model().parameters())
    assert state <= (math.ceil(n / 2)) * 2 * 4  # half of full Adam state


def test_bucketing_reduces_calls(lab):
    out = ldist.run(lab.bucket_call_counts, 2)
    small, big = out[0]
    assert big < small


def test_fsdp2_matches_adam(lab):
    ref = lab.reference_train("adam")
    try:
        out = ldist.run(lab.fsdp2_train, 2)
    except RuntimeError as e:
        if "not supported" in str(e).lower() or "gloo" in str(e).lower():
            pytest.skip(f"FSDP2 unavailable on this backend: {str(e)[:120]}")
        raise
    assert _max_diff(ref, out[0]) < 1e-5
