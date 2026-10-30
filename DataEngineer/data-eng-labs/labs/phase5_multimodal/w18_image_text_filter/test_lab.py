import numpy as np
import pytest
import torch

from common.shapes import pairs


@pytest.fixture(scope="module")
def trained():
    from conftest import load_lab
    return load_lab(__file__).train(n=3000, epochs=5)


def test_loss_is_symmetric_infonce(lab):
    torch.manual_seed(0)
    a = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    scale = torch.tensor(0.0)   # exp(0) = 1
    perfect = lab.clip_loss(a, a, torch.tensor(np.log(50.0)))
    random_ = lab.clip_loss(a, torch.nn.functional.normalize(torch.randn(8, 16), dim=-1), torch.tensor(np.log(50.0)))
    assert perfect < random_
    l = lab.clip_loss(a, a, scale)
    logits = a @ a.t()
    ref = torch.nn.functional.cross_entropy(logits, torch.arange(8))
    assert torch.allclose(l, ref, atol=1e-5)   # symmetric logits: both directions equal


def test_text_encoder_ignores_padding(lab):
    m = lab.TinyCLIP(10)
    a = m.encode_text(torch.tensor([[1, 2, 0, 0]]))
    b = m.encode_text(torch.tensor([[1, 2, 0, 0, 0, 0]]))
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a.norm(dim=-1), torch.ones(1))


def test_zero_shot(lab, trained):
    model, vocab = trained
    imgs, _, labels = pairs(400, seed=9)
    assert lab.zero_shot(model, vocab, imgs, labels) > 0.5   # chance is 1/16


def test_clip_filter_beats_heuristic(lab, trained):
    model, vocab = trained
    imgs, caps, labels = pairs(1500, seed=7, wrong=0.25, junk=0.15)
    s = lab.clip_scores(model, vocab, imgs, caps)
    clip = lab.evaluate(lab.keep_top(s, 0.5), labels)
    heur = lab.evaluate(lab.heuristic_keep(caps), labels)
    assert clip["precision"] > 0.95 > heur["precision"]
    assert lab.keep_top(np.arange(10.0), 0.3).tolist() == [False] * 7 + [True] * 3
