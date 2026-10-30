"""Generate exercise.py from solution.py in every lab directory.

In solution files, the parts the learner must write are fenced like this:

    # >>> SOLUTION: one-line hint shown in the exercise
    ...reference code...
    # <<< SOLUTION

The fence is replaced by the hint as a comment plus
`raise NotImplementedError("TODO(<lab>): <hint>")` at the same indentation.
Everything outside fences (signatures, docstrings, scaffolding, tests) is kept,
which is what makes the codebases "semi-worked".

Usage:  python tools/make_exercises.py            # regenerate all
        python tools/make_exercises.py --check    # fail if any exercise is stale
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OPEN = re.compile(r"^(?P<indent>\s*)(?P<c>#|//) >>> SOLUTION(?::\s*(?P<hint>.*))?$")
CLOSE = re.compile(r"^\s*(#|//) <<< SOLUTION\s*$")
HEADER = (
    "# AUTO-GENERATED from solution.py by tools/make_exercises.py. Edit this file;\n"
    "# fill in every NotImplementedError. Run tests with: LAB=exercise pytest <lab dir>\n"
)


def strip(text: str, lab: str, c_style: bool = False) -> tuple[str, int]:
    out, n, inside = [], 0, False
    for lineno, line in enumerate(text.splitlines(), 1):
        if not inside:
            m = OPEN.match(line)
            if m:
                inside, n = True, n + 1
                ind, hint = m["indent"], (m["hint"] or "implement this").strip()
                if m["c"] == "#":
                    out.append(f"{ind}# TODO: {hint}")
                    out.append(f'{ind}raise NotImplementedError("TODO({lab}): {hint.replace(chr(34), chr(39))}")')
                else:  # C++/CUDA: leave a marker; the tests fail on wrong results
                    out.append(f"{ind}// TODO({lab}): {hint}")
            else:
                out.append(line)
        else:
            if CLOSE.match(line):
                inside = False
            elif OPEN.match(line):
                raise SystemExit(f"{lab}:{lineno}: nested SOLUTION fence")
    if inside:
        raise SystemExit(f"{lab}: unterminated SOLUTION fence")
    header = HEADER if not c_style else HEADER.replace("# ", "// ")
    return header + "\n".join(out) + "\n", n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale, total = [], 0
    sols = [p for ext in ("py", "cpp", "cu") for p in ROOT.glob(f"labs/**/solution*.{ext}")]
    for sol in sorted(sols):
        lab = sol.parent.name
        target = sol.with_name(sol.name.replace("solution", "exercise", 1))
        text, n = strip(sol.read_text(), lab, c_style=sol.suffix in (".cpp", ".cu"))
        total += n
        if args.check:
            if not target.exists() or target.read_text() != text:
                stale.append(str(target.relative_to(ROOT)))
        else:
            target.write_text(text)
    if stale:
        print("stale exercises:\n  " + "\n  ".join(stale))
        return 1
    print(f"{'checked' if args.check else 'wrote'} exercises; {total} TODO blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
