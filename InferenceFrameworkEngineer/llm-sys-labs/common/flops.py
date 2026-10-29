"""FLOP accounting and MFU.

Model FLOPs Utilisation (MFU) = achieved model FLOP/s / hardware peak FLOP/s.
"Model FLOPs" counts only the math the model needs (no recompute), using the
standard estimate from Kaplan et al. / PaLM Appendix B:

    training FLOPs per token  ~= 6 * N  +  12 * n_layer * d_model * seq_len

where N is the non-embedding parameter count, and the second term is
attention's QK^T and AV matmuls (forward + backward).
"""
from __future__ import annotations

# Dense peak TFLOP/s from vendor datasheets (no sparsity). Extend for your GPU.
PEAK_TFLOPS = {
    ("H100-SXM", "bf16"): 989.0,
    ("H100-SXM", "fp8"): 1979.0,
    ("H100-PCIe", "bf16"): 756.0,
    ("A100", "bf16"): 312.0,
    ("L4", "bf16"): 121.0,
    ("RTX4090", "bf16"): 165.2,
}


def train_flops_per_token(n_params_non_embed: int, n_layer: int, d_model: int, seq_len: int) -> float:
    return 6.0 * n_params_non_embed + 12.0 * n_layer * d_model * seq_len


def mfu(tokens_per_sec: float, flops_per_token: float, peak_tflops: float) -> float:
    return tokens_per_sec * flops_per_token / (peak_tflops * 1e12)


def cpu_peak_tflops_estimate(cores: int, ghz: float = 3.0, flops_per_cycle: int = 32) -> float:
    """Rough fp32 peak for an AVX2 FMA core (2 FMA units x 8 lanes x 2 flops).
    Use only to sanity-check CPU MFU; measure your own machine for real numbers."""
    return cores * ghz * 1e9 * flops_per_cycle / 1e12
