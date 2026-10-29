import pytest
import torch


def test_gemm_cpu_program(lab):
    A, B = torch.randn(64, 96), torch.randn(96, 64)
    C = torch.empty(64, 64)
    lab.compile(lab.gemm_cpu(64, 64, 96), "c")(A, B, C)
    torch.testing.assert_close(C, A @ B, rtol=1e-4, atol=1e-4)


def test_blockscaled_gemm(lab):
    torch.manual_seed(0)
    A, B = torch.randn(64, 128), torch.randn(128, 64)
    C, ref = lab.run_blockscaled(A, B)
    torch.testing.assert_close(C, ref, rtol=1e-4, atol=1e-3)
    # FP8 rounding error vs the true product stays small with block scales.
    assert ((C - A @ B).norm() / (A @ B).norm()).item() < 0.05


@pytest.mark.parametrize("causal", [True, False])
def test_attention_cpu_program(lab, causal):
    q, k, v = (torch.randn(2, 24, 16) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(lab.run_attention(q, k, v, causal), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_gemm_gpu_program(lab):
    A = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    B = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    C = torch.empty(256, 256, device="cuda", dtype=torch.float16)
    lab.compile(lab.gemm_gpu(256, 256, 256), "cuda")(A, B, C)
    torch.testing.assert_close(C, A @ B, rtol=1e-2, atol=1e-1)
