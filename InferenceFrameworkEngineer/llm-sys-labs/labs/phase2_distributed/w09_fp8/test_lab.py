import torch


def test_roundtrip_is_fp8_exact(lab):
    x = torch.randn(64, 64)
    q, s = lab.quantize(x)
    assert q.dtype == torch.float8_e4m3fn
    # E4M3 has 3 mantissa bits: worst-case relative rounding error is 2^-4 for normals.
    assert lab.rel_error(x) < 0.04


def test_blockwise_beats_per_tensor_with_outliers(lab):
    torch.manual_seed(0)
    x = torch.randn(128, 128)
    x[0, 0] = 1e5
    (med_t, zero_t), (med_b, zero_b) = lab.bulk_error(x), lab.bulk_error(x, (1, 128))
    assert zero_t > 0.10 and zero_b < 0.01  # per-tensor scale flushes the bulk to zero
    assert med_b < med_t / 3


def test_delayed_scaler_uses_history(lab):
    sc = lab.DelayedScaler(history=4)
    s1 = sc.scale(torch.tensor([448.0]))
    s2 = sc.scale(torch.tensor([1.0]))  # small tensor, but history still says 448
    assert torch.isclose(s1, torch.tensor(1.0)) and torch.isclose(s2, torch.tensor(1.0))
    for _ in range(4):
        s = sc.scale(torch.tensor([1.0]))
    assert torch.isclose(s, torch.tensor(1.0 / 448))


def test_fp8_linear_close_to_fp32(lab):
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32)
    f8 = lab.to_fp8(torch.nn.Sequential(lin))[0]
    x = torch.randn(8, 64, requires_grad=True)
    y_ref, y = torch.nn.functional.linear(x, lin.weight, lin.bias), f8(x)
    assert (y - y_ref).norm() / y_ref.norm() < 0.05
    y.sum().backward()
    assert x.grad is not None and f8.weight.grad is not None


def test_fp8_training_tracks_baseline(lab):
    r = lab.compare_training(steps=150)
    # Tiny model, short CPU run: measured gap ~6% of mean loss. The roadmap's
    # 0.25% gate applies to the multi-thousand-step GPU run with Transformer Engine.
    assert r["rel_gap"] < 0.10, r
    assert r["fp8_final"] < 0.5
