import math

import pytest

from common import dist as ldist


@pytest.mark.parametrize("world", [2, 4])
def test_ring_attention(lab, world):
    out = ldist.run(lab.cp_check, world, "ring")
    assert max(out.values()) < 1e-5


@pytest.mark.parametrize("world", [2, 4])
def test_ulysses(lab, world):
    out = ldist.run(lab.cp_check, world, "ulysses")
    assert max(out.values()) < 1e-5


def test_activation_budget(lab):
    base = lab.activation_bytes_per_layer(4096, 1, 4096, 32)
    assert math.isclose(base, 4096 * 4096 * 34)
    assert math.isclose(lab.activation_bytes_per_layer(4096, 1, 4096, 32, cp=4), base / 4)
    assert math.isclose(lab.activation_bytes_per_layer(4096, 1, 4096, 32, tp=8), base / 8)
    no_flash = lab.activation_bytes_per_layer(4096, 1, 4096, 32, flash=False)
    assert math.isclose(no_flash, 4096 * 4096 * (34 + 5 * 32 * 4096 / 4096))
