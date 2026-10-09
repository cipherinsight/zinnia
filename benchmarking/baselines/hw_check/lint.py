"""Static checks on the handwritten baselines (`target.rs`, `main.nr`).

Errors (each a rule of CONVENTIONS.md or NPB_BASELINES.md):
- `target.rs` not on the shared prelude, or still on halo2-graph;
- `main.nr` without exactly one `// @outputs:` line;
- `main.nr` using the tolerance helpers (`q_eq_with_tol`, `q_close`): a kernel
  `==` is exact;
- `main.nr` with an `unsafe` block not preceded by a `// Safety:` comment;
- NPBench/Pythran: a size constant whose formula does not give its value at
  the fixture's shapes, or one declared below the first function
  (`hw_check.sizes`).

Warnings (to review by hand): every `load_witness` in `target.rs` (a hint must
be pinned, with the argument next to it and a test that wrong hints are
rejected; constants are never witnesses), and every `as uN`/`as iN` cast in
`main.nr` whose line does not show the range check that makes it exact. A cast
of a loop counter or of a value produced by `window`/`index_pos` is fine.
NPBench/Pythran: every number outside the size constants that looks like a
size (a dimension, a product of dimensions, one less or more, their clog2):
it must be a size constant, or not a size.

Usage: python -m hw_check.lint [CASES...] [--npb]
(default: every DS-1000 case, or with --npb every NPBench/Pythran entry with a baseline)
"""
from __future__ import annotations

import json
import re
import sys

from . import sizes
from .entries import Entry, ds1000_entries, npb_entries, resolve, with_baselines

CAST = re.compile(r"\bas\s+[ui](8|16|32|64|128)\b")


def lint_target(path) -> tuple[list[str], list[str]]:
    lines = path.read_text().splitlines()
    src = "\n".join(lines)
    errors = []
    if "use crate::prelude::*;" not in src:
        errors.append("does not use the shared prelude")
    if "halo2_graph" in src:
        errors.append("still uses halo2-graph")
    warnings = [f"line {i + 1}: hint `{l.strip()}`" for i, l in enumerate(lines) if "load_witness" in l]
    return errors, warnings


def lint_noir(path) -> tuple[list[str], list[str]]:
    lines = path.read_text().splitlines()
    errors, warnings = [], []
    if sum(1 for l in lines if l.strip().startswith("// @outputs:")) != 1:
        errors.append("needs exactly one `// @outputs:` line")
    for i, line in enumerate(lines):
        code = line.split("//", 1)[0]
        if "q_eq_with_tol" in code or "q_close" in code:
            errors.append(f"line {i + 1}: tolerance equality")
        if "unsafe" in code and not any("Safety:" in l for l in lines[max(0, i - 3):i]):
            errors.append(f"line {i + 1}: unsafe without a `// Safety:` comment")
        if CAST.search(code):
            warnings.append(f"line {i + 1}: cast `{code.strip()}`")
    return errors, warnings


def lint_entry(entry: Entry) -> list[str]:
    report = []
    for path, lint in ((entry.target_rs, lint_target), (entry.main_nr, lint_noir)):
        if not path.exists():
            continue
        errors, warnings = lint(path)
        if entry.function is not None:  # shape-generic: sizes are named constants
            size_errors, size_warnings = sizes.check(path, json.loads(entry.fixture.read_text()))
            errors, warnings = errors + size_errors, warnings + size_warnings
        report += [f"ERROR {path.name}: {e}" for e in errors] + [f"warn  {path.name}: {w}" for w in warnings]
    return report


def main() -> int:
    args = sys.argv[1:]
    npb = "--npb" in args
    names = [a for a in args if a != "--npb"]
    if names:
        entries = [resolve(n) for n in names]
    else:
        entries = with_baselines(npb_entries()) if npb else list(ds1000_entries().values())
    n_errors = 0
    for entry in entries:
        report = lint_entry(entry)
        n_errors += sum(1 for r in report if r.startswith("ERROR"))
        for r in report:
            print(f"{entry.case}: {r}")
    print(f"{n_errors} errors")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
