# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;
# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>
"""Week 3: PyTorch internals.

Roadmap: "Write a custom autograd function and a C++/CUDA extension. Use
torch.profiler and memory snapshots to find the allocator's fragmentation."
(The C++/CUDA extension mechanics are Week 2's `load`; here we register the op
properly with the dispatcher so torch.compile can see it.)

Parts:
  1. RMSNormFn       - autograd.Function with a hand-derived backward
  2. rmsnorm op      - torch.library.custom_op + fake (meta) kernel + autograd,
                       so it composes with torch.compile and opcheck
  3. saved_activation_bytes - count what autograd keeps alive (memory budgeting)
  4. top_ops         - torch.profiler: where does time go?
  5. graph_breaks    - torch._dynamo.explain: why compile is not speeding you up

Everything runs on CPU and CUDA.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch
from torch.utils.checkpoint import checkpoint

PREFIX = pathlib.Path(__file__).stem


# ------------------------------------------------------------------ part 1

class RMSNormFn(torch.autograd.Function):
    """y = x * rstd * w,  rstd = (mean(x^2, -1) + eps)^(-1/2)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
        # TODO: compute rstd over the last dim, save (x, w, rstd) for backward, return y
        raise NotImplementedError("TODO(w03_pytorch_internals): compute rstd over the last dim, save (x, w, rstd) for backward, return y")

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        # TODO: gw = g*w; dx = rstd*gw - x*rstd^3*mean(gw*x); dw = sum over leading dims of g*x*rstd
        raise NotImplementedError("TODO(w03_pytorch_internals): gw = g*w; dx = rstd*gw - x*rstd^3*mean(gw*x); dw = sum over leading dims of g*x*rstd")


def rmsnorm_fn(x, w, eps: float = 1e-6):
    return RMSNormFn.apply(x, w, eps)


# ------------------------------------------------------------------ part 2
# A namespaced op per file so exercise and solution can coexist in one process.

OP_NS = f"w03_{PREFIX}"


@torch.library.custom_op(f"{OP_NS}::rmsnorm", mutates_args=())
def rmsnorm_op(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # Real kernel. On GPU you would dispatch to your Triton/CUDA kernel here.
    return x * (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt() * w


@rmsnorm_op.register_fake
def _(x, w, eps):
    # TODO: a fake (meta) kernel returns an empty tensor with the right shape/dtype/device
    raise NotImplementedError("TODO(w03_pytorch_internals): a fake (meta) kernel returns an empty tensor with the right shape/dtype/device")


def _setup_ctx(ctx, inputs, output):
    x, w, eps = inputs
    ctx.save_for_backward(x, w)
    ctx.eps = eps


def _backward(ctx, g):
    x, w = ctx.saved_tensors
    rstd = (x.pow(2).mean(-1, keepdim=True) + ctx.eps).rsqrt()
    gw = g * w
    dx = rstd * gw - x * rstd.pow(3) * (gw * x).mean(-1, keepdim=True)
    dw = (g * x * rstd).reshape(-1, x.shape[-1]).sum(0)
    return dx, dw, None


rmsnorm_op.register_autograd(_backward, setup_context=_setup_ctx)


# ------------------------------------------------------------------ part 3

def saved_activation_bytes(fn, *inputs) -> int:
    """Bytes of tensors autograd saves for backward while running fn(*inputs).

    Uses saved_tensors_hooks, which see every tensor packed for backward. Count
    each storage once (views share storage)."""
    seen: dict[int, int] = {}

    # TODO: pack hook records storage ptr -> nbytes and returns the tensor; unpack returns it
    raise NotImplementedError("TODO(w03_pytorch_internals): pack hook records storage ptr -> nbytes and returns the tensor; unpack returns it")
    del out
    # Parameters are also "saved" but they live regardless; exclude them.
    params = set()
    for x in inputs:
        if isinstance(x, torch.nn.Module):
            params |= {p.untyped_storage().data_ptr() for p in x.parameters()}
    return sum(v for k, v in seen.items() if k not in params)


def mlp_stack(depth: int = 4, d: int = 256) -> torch.nn.Module:
    layers = []
    for _ in range(depth):
        layers += [torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d)]
    return torch.nn.Sequential(*layers)


def run_plain(model, x):
    return model(x).sum()


def run_checkpointed(model, x):
    """Activation checkpointing: keep only block inputs, recompute inside."""
    # TODO: wrap each (Linear, GELU, Linear) triple in torch.utils.checkpoint.checkpoint(use_reentrant=False)
    raise NotImplementedError("TODO(w03_pytorch_internals): wrap each (Linear, GELU, Linear) triple in torch.utils.checkpoint.checkpoint(use_reentrant=False)")


# ------------------------------------------------------------------ part 4

def top_ops(fn, n: int = 5) -> list[tuple[str, float]]:
    """Run fn under torch.profiler and return the n ops with most self time (us)."""
    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    # TODO: profile fn(), then sort key_averages() by self CPU (+ device) time
    raise NotImplementedError("TODO(w03_pytorch_internals): profile fn(), then sort key_averages() by self CPU (+ device) time")


# ------------------------------------------------------------------ part 5

def graph_breaks(fn, *args) -> int:
    """Number of dynamo graph breaks when compiling fn(*args)."""
    import torch._dynamo as dynamo

    dynamo.reset()
    # TODO: torch._dynamo.explain(fn)(*args).graph_break_count
    raise NotImplementedError("TODO(w03_pytorch_internals): torch._dynamo.explain(fn)(*args).graph_break_count")


def leaky_fn(x):
    """Contains a classic graph break: a data-dependent Python branch on .item()."""
    y = x * 2
    if y.sum().item() > 0:
        y = y + 1
    return y.relu()


def fixed_fn(x):
    """Same math with no host sync: the branch becomes a tensor op."""
    # TODO: replace the Python if with torch.where on the reduced value
    raise NotImplementedError("TODO(w03_pytorch_internals): replace the Python if with torch.where on the reduced value")


if __name__ == "__main__":
    m = mlp_stack()
    x = torch.randn(64, 256, requires_grad=True)
    print("saved bytes plain        ", saved_activation_bytes(run_plain, m, x))
    print("saved bytes checkpointed ", saved_activation_bytes(run_checkpointed, m, x))
    print("graph breaks leaky/fixed ", graph_breaks(leaky_fn, x.detach()), graph_breaks(fixed_fn, x.detach()))
    for name, t in top_ops(lambda: run_plain(m, x).backward()):
        print(f"{t:10.1f} us  {name}")
