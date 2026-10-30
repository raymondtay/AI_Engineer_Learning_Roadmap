import numpy as np
import pytest


def test_psi_properties(lab):
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert lab.psi(a, a) == pytest.approx(0, abs=1e-9)
    assert lab.psi(a, rng.normal(0, 1, 5000)) < 0.02
    small, big = lab.psi(a, rng.normal(0.2, 1, 5000)), lab.psi(a, rng.normal(1.0, 1, 5000))
    assert 0.02 < small < 0.1 < 0.25 < big
    assert lab.psi_categorical({"en": 0.5, "de": 0.5}, {"en": 0.5, "de": 0.5}) == pytest.approx(0)


@pytest.fixture(scope="module")
def ref():
    from conftest import load_lab
    lab = load_lab(__file__)
    return lab.profile(lab.clean_batch(0, n=2000))


def test_no_false_alarms_on_clean(lab, ref):
    for s in range(3):
        assert lab.check(ref, lab.profile(lab.clean_batch(100 + s, n=1000))) == []


@pytest.mark.parametrize("name", ["broken extractor", "truncating fetcher", "language-ID swap"])
def test_detects_injected_failure(lab, ref, name):
    alerts = lab.check(ref, lab.profile(lab.INJECTIONS[name](300, n=1000)))
    assert alerts, name
    if name == "language-ID swap":
        assert any(a.startswith("lang") for a in alerts)
