"""Device and backend selection shared by every lab.

Every lab must run on CPU. When a CUDA GPU is present the same code paths
run on it, and GPU-only labs (CUDA kernels, Transformer Engine, Megatron-Core,
vLLM, SGLang, verl, slime) are enabled.

Environment overrides:
  LAB_DEVICE=cpu|cuda      force a device
  TRITON_INTERPRET=1       run Triton kernels on CPU through the interpreter
"""
from __future__ import annotations

import os
import importlib.util

import torch


def device() -> torch.device:
    forced = os.environ.get("LAB_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_cuda() -> bool:
    return device().type == "cuda"


def autocast_dtype() -> torch.dtype:
    """BF16 everywhere it is supported. CPU autocast supports bfloat16 too."""
    if is_cuda() and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


def dist_backend() -> str:
    """NCCL on GPU, gloo on CPU. Same collectives API either way."""
    return "nccl" if is_cuda() else "gloo"


def sm_version() -> int:
    """Compute capability as an int, e.g. 90 for Hopper, 0 on CPU."""
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def triton_device() -> str:
    """Where Triton kernels execute. On CPU this requires TRITON_INTERPRET=1,
    which must be set before `import triton`; common/__init__ handles that."""
    return "cuda" if is_cuda() else "cpu"


def describe() -> str:
    d = device()
    parts = [f"device={d}", f"torch={torch.__version__}"]
    if d.type == "cuda":
        parts.append(f"gpu={torch.cuda.get_device_name()} sm_{sm_version()}")
    parts.append(f"triton_interpret={os.environ.get('TRITON_INTERPRET', '0')}")
    return " ".join(parts)
