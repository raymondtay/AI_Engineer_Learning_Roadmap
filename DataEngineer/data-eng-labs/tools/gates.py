"""Collect every lab's gate measurement into GATES.md - the portfolio's single page.

Each lab exposes report(), which runs its measurement and returns a markdown
table. This script runs them all against your exercise code (default) or the
reference solutions (LAB=solution). Labs with unfinished TODOs show as
"not implemented". Optional toolchains (Spark, cargo) degrade gracefully.

Usage:  python tools/gates.py                  # your work -> GATES.md
        LAB=solution python tools/gates.py     # reference numbers for this machine
        python tools/gates.py w07 w13          # only some weeks
Full run on a 2-core CPU: about 8 minutes (Week 24 alone is ~4).
"""
from __future__ import annotations

import contextlib
import datetime
import importlib.util
import io
import os
import pathlib
import platform
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WHICH = os.environ.get("LAB", "exercise")


def labs():
    return sorted(p for p in (ROOT / "labs").glob("phase*/w*") if p.is_dir())


def run(lab_dir: pathlib.Path) -> tuple[str, float]:
    path = lab_dir / f"{WHICH}.py"
    if not path.exists():
        return f"_missing {path.name} (run tools/make_exercises.py)_", 0.0
    name = f"gate_{lab_dir.name}_{WHICH}"
    t0 = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        with contextlib.redirect_stderr(io.StringIO()):
            out = mod.report()
        if isinstance(out, list) and out and isinstance(out[0], dict):   # rows -> markdown table
            from common.bench import table
            out = table(out, list(out[0]))
        elif isinstance(out, dict):
            from common.bench import table
            out = table([{"metric": k, "value": v} for k, v in out.items()], ["metric", "value"])
        out = str(out)
    except NotImplementedError as e:
        out = f"_not implemented yet: {e}_"
    except Exception:
        out = "```\n" + traceback.format_exc(limit=3) + "```"
    return out, time.perf_counter() - t0


def main(argv: list[str]) -> int:
    chosen = [l for l in labs() if not argv or any(l.name.startswith(a) for a in argv)]
    try:
        import torch
        dev = "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    except Exception:
        dev = "torch not installed"
    lines = [f"# Gates ({WHICH})", "",
             f"{datetime.datetime.now():%Y-%m-%d %H:%M} | {platform.platform()} | Python {platform.python_version()} | "
             f"{os.cpu_count()} cores | {dev}", ""]
    dest = ROOT / ("GATES.md" if WHICH == "exercise" else f"GATES_{WHICH}.md")
    for lab in chosen:
        print(f"... {lab.parent.name}/{lab.name}", flush=True)
        out, secs = run(lab)
        lines += [f"## {lab.parent.name} / {lab.name}  ({secs:.1f}s)", "", out, ""]
        dest.write_text("\n".join(lines))   # write as we go: a crash keeps finished gates
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
