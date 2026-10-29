import os

# Triton reads TRITON_INTERPRET at import time. Without a GPU, turn the
# interpreter on before any lab imports triton so kernels run on CPU tensors.
try:
    import torch as _torch

    if not _torch.cuda.is_available() and os.environ.get("LAB_DEVICE", "") != "cuda":
        os.environ.setdefault("TRITON_INTERPRET", "1")
except ImportError:  # torch missing: let the lab fail with a clear message
    pass
