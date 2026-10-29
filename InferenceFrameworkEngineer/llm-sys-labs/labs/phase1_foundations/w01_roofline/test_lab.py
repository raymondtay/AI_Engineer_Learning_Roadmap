import math


def test_gemm_square_fp16(lab):
    # 2n^3 / (3 n^2 * 2 bytes) = n / 3
    assert math.isclose(lab.ai_gemm(4096, 4096, 4096, 2), 4096 / 3, rel_tol=1e-9)


def test_gemm_skinny_is_low_intensity(lab):
    # A decode-time GEMV (M=1) is memory-bound on any modern GPU.
    assert lab.ai_gemm(1, 4096, 4096, 2) < 2.0


def test_softmax_and_layernorm(lab):
    assert math.isclose(lab.ai_softmax(8, 1024, 2), 1.25)
    # 8*n / ((2n + 2*cols) * 2B) with n = 8*1024 -> 16/9; gamma/beta reads pull it below 2
    assert math.isclose(lab.ai_layernorm(8, 1024, 2), 16 / 9)


def test_ridge_and_roofline(lab):
    h100_bf16, h100_bw = 989e12, 3.35e12  # datasheet dense BF16, HBM3 bandwidth
    assert math.isclose(lab.ridge_point(h100_bf16, h100_bw), 989 / 3.35)
    t, b = lab.roofline_time(2 * 4096**3, 3 * 4096**2 * 2, h100_bf16, h100_bw)
    assert b == "compute" and math.isclose(t, 2 * 4096**3 / h100_bf16)
    t, b = lab.roofline_time(5 * 2**20, 2 * 2**20 * 2, h100_bf16, h100_bw)
    assert b == "memory"


def test_report_runs(lab):
    out = lab.report(n=256)
    assert "gemm" in out and "softmax" in out
