"""Pytest wiring shared by all labs.

Each lab's test file asks for the `lab` fixture, which imports either
exercise.py (default: your work) or solution.py (LAB=solution) from the same
directory. Markers:
  @pytest.mark.spark  needs pyspark + Java; skipped otherwise
  @pytest.mark.rust   needs cargo; skipped otherwise
  @pytest.mark.slow   longer runs; skipped unless RUN_SLOW=1
"""
import importlib.util
import os
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    for m in ("spark: needs pyspark and Java", "rust: needs cargo", "slow: long-running; set RUN_SLOW=1"):
        config.addinivalue_line("markers", m)


def _has_spark():
    return importlib.util.find_spec("pyspark") is not None and shutil.which("java") is not None


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "spark" in item.keywords and not _has_spark():
            item.add_marker(pytest.mark.skip(reason="needs pyspark + Java"))
        if "rust" in item.keywords and shutil.which("cargo") is None:
            item.add_marker(pytest.mark.skip(reason="needs cargo (Rust)"))
        if "slow" in item.keywords and os.environ.get("RUN_SLOW") != "1":
            item.add_marker(pytest.mark.skip(reason="set RUN_SLOW=1"))


def load_lab(test_file: str, stem: str | None = None):
    which = os.environ.get("LAB", "exercise")
    d = pathlib.Path(test_file).resolve().parent
    path = d / f"{which}{'_' + stem if stem else ''}.py"
    if not path.exists():
        pytest.fail(f"{path} missing; run: python tools/make_exercises.py")
    mod_name = f"lab_{d.name}_{which}_{stem or 'main'}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def lab(request):
    return load_lab(str(request.fspath))
