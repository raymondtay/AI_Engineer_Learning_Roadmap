import torch

from common.device import device


def test_rmsnorm_gradcheck(lab):
    x = torch.randn(3, 5, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: lab.rmsnorm_fn(a, b, 1e-6), (x, w))


def test_rmsnorm_matches_reference(lab):
    x = torch.randn(4, 16, device=device())
    w = torch.randn(16, device=device())
    ref = torch.nn.functional.rms_norm(x, (16,), w, eps=1e-6)
    torch.testing.assert_close(lab.rmsnorm_fn(x, w), ref, rtol=1e-5, atol=1e-5)


def test_custom_op_opcheck(lab):
    x = torch.randn(4, 16, requires_grad=True)
    w = torch.randn(16, requires_grad=True)
    torch.library.opcheck(lab.rmsnorm_op, (x, w, 1e-6))


def test_checkpointing_saves_memory(lab):
    m = lab.mlp_stack(depth=4, d=128)
    x = torch.randn(32, 128, requires_grad=True)
    plain = lab.saved_activation_bytes(lab.run_plain, m, x)
    ckpt = lab.saved_activation_bytes(lab.run_checkpointed, m, x)
    assert plain > 0 and ckpt < plain / 3
    # Gradients must be identical with and without checkpointing.
    g1 = torch.autograd.grad(lab.run_plain(m, x), x)[0]
    g2 = torch.autograd.grad(lab.run_checkpointed(m, x), x)[0]
    torch.testing.assert_close(g1, g2)


def test_profiler_finds_matmul(lab):
    a = torch.randn(256, 256)
    rows = lab.top_ops(lambda: (a @ a).relu())
    assert any("mm" in name for name, _ in rows)


def test_graph_breaks(lab):
    x = torch.randn(8)
    assert lab.graph_breaks(lab.leaky_fn, x) >= 1
    assert lab.graph_breaks(lab.fixed_fn, x) == 0
    torch.testing.assert_close(lab.fixed_fn(x), lab.leaky_fn(x))
