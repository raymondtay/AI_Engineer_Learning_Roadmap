import math
from dataclasses import replace

import pytest
import torch


def test_manual_attention_matches_sdpa(lab):
    torch.manual_seed(0)
    cfg = lab.Config()
    attn = lab.CausalSelfAttention(cfg)
    x = torch.randn(2, 10, cfg.d_model)
    torch.testing.assert_close(attn(x, use_sdpa=False), attn(x, use_sdpa=True), rtol=1e-4, atol=1e-5)


def test_lr_schedule(lab):
    cfg = lab.Config(lr=1.0, warmup=10, steps=110)
    assert math.isclose(lab.lr_at(0, cfg), 0.1)
    assert math.isclose(lab.lr_at(9, cfg), 1.0)
    assert math.isclose(lab.lr_at(110, cfg), 0.1, abs_tol=1e-9)
    assert lab.lr_at(30, cfg) > lab.lr_at(80, cfg)


def test_weight_decay_groups(lab):
    model = lab.GPT(lab.Config())
    opt = lab.make_optimizer(model, lab.Config())
    wd = {g["weight_decay"] for g in opt.param_groups}
    assert wd == {0.1, 0.0}
    no_decay = [p for g in opt.param_groups if g["weight_decay"] == 0 for p in g["params"]]
    assert all(p.dim() < 2 for p in no_decay)


def test_training_learns(lab):
    r = lab.train(lab.Config(steps=150))
    assert r["losses"][0] > 4.0 and r["final_loss"] < 1.5


def test_checkpointing_same_gradients(lab):
    torch.manual_seed(0)
    cfg = lab.Config()
    m = lab.GPT(cfg)
    x = torch.randint(0, 256, (2, cfg.seq_len))
    _, l1 = m(x, x)
    g1 = torch.autograd.grad(l1, list(m.parameters()))
    m.cfg = replace(cfg, checkpoint=True)
    _, l2 = m(x, x)
    g2 = torch.autograd.grad(l2, list(m.parameters()))
    for a, b in zip(g1, g2):
        torch.testing.assert_close(a, b)


def test_bf16_trains(lab):
    r = lab.train(lab.Config(steps=150, bf16=True))
    assert r["final_loss"] < 2.0


@pytest.mark.slow
def test_compile_trains(lab):
    r = lab.train(lab.Config(steps=60, compile=True))
    assert r["final_loss"] < 3.0
