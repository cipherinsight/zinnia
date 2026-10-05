"""Static checks on the handwritten baselines (`target.rs`, `main.nr`).

Errors (each a rule of CONVENTIONS.md):
- `target.rs` not on the shared prelude, or still on halo2-graph;
- `target.rs` calling `load_witness` (constants are circuit constants and
  hints live in the q32 gadgets, where they are pinned and tested);
- `main.nr` without exactly one `// @outputs:` line;
- `main.nr` using the tolerance helpers (`q_eq_with_tol`, `q_close`): a kernel
  `==` is exact;
- `main.nr` with an `unsafe` block not preceded by a `// Safety:` comment.

Warnings (to review by hand): every `as uN`/`as iN` cast in `main.nr` whose
line does not show the range check that makes it exact. A cast of a loop
counter or of a value produced by `window`/`index_pos` is fine.

Usage: python -m hw_check.lint [CASES...]   (default: every case)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DATASET = Path(__file__).resolve().parents[2] / "datasets" / "ds1000"
CAST = re.compile(r"\bas\s+[ui](8|16|32|64|128)\b")


def lint_target(path: Path) -> list[str]:
    src = path.read_text()
    errors = []
    if "use crate::prelude::*;" not in src:
        errors.append("does not use the shared prelude")
    if "halo2_graph" in src:
        errors.append("still uses halo2-graph")
    if "load_witness" in src:
        errors.append("calls load_witness directly")
    return errors


def lint_noir(path: Path) -> tuple[list[str], list[str]]:
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


def main() -> int:
    names = [a if a.startswith("ds1000_case") else f"ds1000_case{a}" for a in sys.argv[1:]]
    dirs = [DATASET / n for n in names] if names else sorted(DATASET.iterdir())
    n_errors = 0
    for d in dirs:
        report = []
        if (d / "target.rs").exists():
            report += [f"ERROR target.rs: {e}" for e in lint_target(d / "target.rs")]
        if (d / "main.nr").exists():
            errors, warnings = lint_noir(d / "main.nr")
            report += [f"ERROR main.nr: {e}" for e in errors] + [f"warn  main.nr: {w}" for w in warnings]
        n_errors += sum(1 for r in report if r.startswith("ERROR"))
        for r in report:
            print(f"{d.name}: {r}")
    print(f"{n_errors} errors")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
