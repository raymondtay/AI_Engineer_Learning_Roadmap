import torch

from common.device import device


def test_softmax(lab):
    x = torch.randn(17, 100, device=device())  # non power-of-2 width exercises masking
    torch.testing.assert_close(lab.softmax(x), torch.softmax(x, -1), rtol=1e-5, atol=1e-6)


def test_rmsnorm_forward_backward(lab):
    x = torch.randn(4, 6, 48, device=device(), requires_grad=True)
    w = torch.randn(48, device=device(), requires_grad=True)
    y = lab.rmsnorm(x, w)
    ref = torch.nn.functional.rms_norm(x, (48,), w, eps=1e-6)
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)
    g = torch.randn_like(y)
    dx, dw = torch.autograd.grad(y, (x, w), g)
    rdx, rdw = torch.autograd.grad(ref, (x, w), g)
    torch.testing.assert_close(dx, rdx, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dw, rdw, rtol=1e-4, atol=1e-4)


def test_matmul(lab):
    a = torch.randn(70, 50, device=device())
    b = torch.randn(50, 90, device=device())
    torch.testing.assert_close(lab.matmul(a, b), a @ b, rtol=1e-3, atol=1e-3)
