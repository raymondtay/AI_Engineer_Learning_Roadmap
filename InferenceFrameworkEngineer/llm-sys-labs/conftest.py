"""Pytest wiring shared by all labs.

Each lab's test file asks for the `lab` fixture, which imports either
exercise.py (default: your work) or solution.py (LAB=solution) from the same
directory. Markers:
  @pytest.mark.gpu   needs a CUDA device; skipped on CPU
  @pytest.mark.slow  longer runs; skipped unless RUN_SLOW=1
"""
import importlib.util
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import common  # noqa: E402,F401  (sets TRITON_INTERPRET on CPU before triton import)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")
    config.addinivalue_line("markers", "slow: long-running; set RUN_SLOW=1")


def pytest_collection_modifyitems(config, items):
    import torch

    no_gpu = pytest.mark.skip(reason="needs CUDA GPU")
    no_slow = pytest.mark.skip(reason="set RUN_SLOW=1")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(no_gpu)
        if "slow" in item.keywords and os.environ.get("RUN_SLOW") != "1":
            item.add_marker(no_slow)


def load_lab(test_file: str, stem: str | None = None):
    """Import exercise or solution module that sits next to test_file."""
    which = os.environ.get("LAB", "exercise")
    d = pathlib.Path(test_file).resolve().parent
    name = f"{which}{'_' + stem if stem else ''}.py"
    path = d / name
    if not path.exists():
        pytest.fail(f"{path} missing; run: python tools/make_exercises.py")
    mod_name = f"lab_{d.name}_{which}_{stem or 'main'}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod  # needed for pickling across spawned processes
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def lab(request):
    return load_lab(str(request.fspath))
