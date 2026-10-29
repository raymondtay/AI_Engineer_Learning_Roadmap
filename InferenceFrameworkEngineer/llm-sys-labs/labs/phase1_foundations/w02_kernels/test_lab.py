import torch

from common.device import device


def test_vector_add(lab):
    a, b = torch.randn(100_003, device=device()), torch.randn(100_003, device=device())
    torch.testing.assert_close(lab.vector_add(a, b), a + b)


def test_reduce_sum(lab):
    x = torch.randn(1_000_003, device=device())
    torch.testing.assert_close(lab.reduce_sum(x).cpu(), x.double().sum().float().cpu(), rtol=1e-4, atol=1e-2)


def test_sgemm_odd_shapes(lab):
    # Non-multiples of the tile size exercise the edge padding.
    a = torch.randn(130, 77, device=device())
    b = torch.randn(77, 101, device=device())
    torch.testing.assert_close(lab.sgemm(a, b), a @ b, rtol=1e-4, atol=1e-3)


def test_sgemm_square(lab):
    a = torch.randn(256, 256, device=device())
    b = torch.randn(256, 256, device=device())
    torch.testing.assert_close(lab.sgemm(a, b), a @ b, rtol=1e-4, atol=1e-3)
