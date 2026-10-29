"""Multi-process launcher for distributed labs.

On CPU it starts `world_size` processes on one machine with the gloo backend.
On GPU it uses NCCL with one process per device. The lab function gets
(rank, world_size, *args) and the process group is already initialised.

On a real cluster use `torchrun` instead; every distributed lab also exposes a
`main()` that reads RANK/WORLD_SIZE from the environment for that purpose.
"""
from __future__ import annotations

import io
import os
import socket
import traceback
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from common.device import dist_backend


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve(ref):
    """Re-import the lab function inside a spawned child.

    Labs are loaded from file paths (exercise.py / solution.py), so the child
    cannot import them by module name. We ship (module name, file, function
    name) and load the file again in the child."""
    import importlib.util
    import sys

    mod_name, path, fn_name = ref
    if mod_name not in sys.modules:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return getattr(sys.modules[mod_name], fn_name)


def _ref(fn):
    import sys

    mod = sys.modules[fn.__module__]
    return (fn.__module__, os.path.abspath(mod.__file__), fn.__name__)


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return obj


def _worker(rank, world_size, port, ref, args, q):
    fn = _resolve(ref)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    backend = dist_backend()
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        out = fn(rank, world_size, *args)
        # Serialise to bytes: tensors put on a Queue directly are shared-memory
        # handles that vanish when this process exits.
        buf = io.BytesIO()
        torch.save(_to_cpu(out), buf)
        q.put((rank, "ok", buf.getvalue()))
    except BaseException:  # report to parent instead of hanging
        q.put((rank, "err", traceback.format_exc()))
    finally:
        # No barrier here: if one rank failed, a barrier would hang the others.
        if dist.is_initialized():
            dist.destroy_process_group()


def run(fn: Callable, world_size: int, *args):
    """Run fn on world_size ranks; return {rank: result}. Raises on any failure."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    ref = _ref(fn)
    procs = [ctx.Process(target=_worker, args=(r, world_size, port, ref, args, q)) for r in range(world_size)]
    for p in procs:
        p.start()
    results, errors = {}, []
    for _ in range(world_size):
        rank, status, payload = q.get(timeout=600)
        if status == "ok":
            results[rank] = torch.load(io.BytesIO(payload), weights_only=False)
        else:
            last = payload.strip().splitlines()[-1] if payload.strip() else "unknown error"
            errors.append(f"rank {rank}: {last}\n{payload}")
    for p in procs:
        p.join(timeout=60)
    if errors:
        raise RuntimeError("\n".join(errors))
    return results


def device_for_rank(rank: int) -> torch.device:
    if dist_backend() == "nccl":
        return torch.device("cuda", rank % torch.cuda.device_count())
    return torch.device("cpu")
