import pytest
import torch

from common.device import device

TOL = dict(rtol=2e-3, atol=2e-3)  # TF32 on GPU; exact fp32 under the CPU interpreter


@pytest.mark.parametrize("causal", [True, False])
def test_dense_forward_backward(lab, causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, 64, 32, device=device(), requires_grad=True) for _ in range(3))
    o = lab.attention(q, k, v, causal)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(o, ref, **TOL)
    g = torch.randn_like(o)
    ours = torch.autograd.grad(o, (q, k, v), g)
    theirs = torch.autograd.grad(ref, (q, k, v), g)
    for a, b in zip(ours, theirs):
        torch.testing.assert_close(a, b, **TOL)


def test_varlen_packed(lab):
    torch.manual_seed(1)
    cu = torch.tensor([0, 37, 100, 164], dtype=torch.int32, device=device())  # lengths 37, 63, 64
    q, k, v = (torch.randn(164, 2, 32, device=device(), requires_grad=True) for _ in range(3))
    o = lab.flash_attn_varlen(q, k, v, cu, 64)
    ref = lab.reference_varlen(q, k, v, cu)
    torch.testing.assert_close(o, ref, **TOL)
    g = torch.randn_like(o)
    for a, b in zip(torch.autograd.grad(o, (q, k, v), g), torch.autograd.grad(ref, (q, k, v), g)):
        torch.testing.assert_close(a, b, **TOL)


@pytest.mark.gpu
def test_speed_gate_reported(lab):
    print(lab.benchmark_gpu())
