"""Size constants of the NPBench/Pythran baselines.

The NPBench/Pythran kernels are shape-generic: the performance set runs them at
the fixture's shapes, and the scaling study at larger ones. So that a baseline
can be instantiated at other shapes by rewriting numbers alone, every number
in it that is derived from the input shapes is a named constant declared at
the top of the file, with the formula that derives it from the fixture's
dimensions (rule in `NPB_BASELINES.md`, section "Size constants"):

    const N: usize = 5; // = M_0                     (target.<f>.rs)
    global N: u32 = 5; // = M_0                      (main.<f>.nr)
    const IDX_BITS: usize = 3; // = clog2(M_0)

A dimension is named `<parameter>_<axis>` (axis 0 first), the names of the
scaling study's rewrite checker (`benchmarking/scaling/rewrite.py`). A formula
may use dimensions, integer constants, `+ - * // % ** <<` and the functions
`clog2` (ceil log2: the bits of an index into n positions), `bits` (the bit
length of n), `cdiv`, `max` and `min`. Constants derived from other constants
by the language itself (`const NN: usize = N * N;`, `global LAST: u32 = N - 1;`)
need no formula. A Noir baseline's `// @outputs:` line names its counts by
these constants too (`float[N * N]`; `hw_check.noir_driver` evaluates them).

This module checks that every declared formula gives the declared value at the
fixture's shapes, and lists numbers outside the declarations that look like
sizes (for `hw_check.lint`). With `--fixture`, it prints the baseline
instantiated at that fixture's shapes: each size constant's literal replaced
by its formula's value, nothing else changed.

Usage:
    python -m hw_check.sizes <case> [--file target.rs|main.nr] [--fixture SCALED.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: `const NAME: T = 5; // = formula` (Rust) or `global NAME: T = 5; // = formula` (Noir).
DECLARATION = re.compile(
    r"^(?P<head>\s*(?:pub\s+)?(?:const|global)\s+(?P<name>[A-Z][A-Z0-9_]*)\s*:\s*[^=]+=\s*)"
    r"(?P<value>\d+)(?P<tail>\s*;\s*//\s*=\s*(?P<formula>.+?)\s*)$")
INTEGER = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
#: `noir_hw::array_index::<8>` / `index_pos::<28>`: the width follows the array length.
INDEX_WIDTH = re.compile(r"\b(array_index|index_pos)::<\d+>")
#: Widths fixed by the number format (Q32, the 64-bit window), never sizes.
FORMAT_WIDTHS = {32, 63, 64}


def clog2(n: int) -> int:
    if n < 1:
        raise ValueError(f"clog2({n})")
    return (n - 1).bit_length()


def bits(n: int) -> int:
    return int(n).bit_length()


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


FUNCTIONS = {"clog2": clog2, "bits": bits, "cdiv": cdiv, "max": max, "min": min}
_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift)


def evaluate(formula: str, dims: dict[str, int]) -> int:
    """The value of a size formula; anything but the allowed syntax is an error."""
    tree = ast.parse(formula, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id not in dims and node.id not in FUNCTIONS:
                raise ValueError(f"unknown name {node.id!r} (dimensions: {', '.join(sorted(dims))})")
        elif isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS) or node.keywords:
                raise ValueError(f"only {sorted(FUNCTIONS)} may be called")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _OPERATORS):
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, int) or isinstance(node.value, bool):
                raise ValueError(f"constant {node.value!r} is not an integer")
        elif not isinstance(node, (ast.Expression, ast.Load, ast.UnaryOp, ast.USub, *_OPERATORS)):
            raise ValueError(f"{type(node).__name__} is not allowed")
    value = eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, {**FUNCTIONS, **dims})  # noqa: S307
    if not isinstance(value, int):
        raise ValueError(f"value {value!r} is not an integer")
    return value


def dimensions(fixture: dict) -> dict[str, int]:
    """`<parameter>_<axis>` -> length, for every array parameter of a fixture."""
    return {f"{p['name']}_{axis}": n for p in fixture["parameters"] if p["kind"] == "array"
            for axis, n in enumerate(p["shape"])}


def size_like(dims: dict[str, int], fixture: dict) -> set[int]:
    """Numbers a hard-coded size would likely be: each dimension, one less and
    one more, the product of consecutive dimensions of a parameter, and their
    `clog2`; only those of at least 3 (smaller numbers are too common)."""
    values: set[int] = set()
    for p in fixture["parameters"]:
        shape = p.get("shape", []) if p["kind"] == "array" else []
        for i in range(len(shape)):
            for j in range(i + 1, len(shape) + 1):
                n = math.prod(shape[i:j])
                values |= {n, n - 1, n + 1, clog2(n)} if n >= 1 else set()
    return {v for v in values if v >= 3} - FORMAT_WIDTHS


@dataclass
class SizeConstant:
    line: int
    name: str
    value: int
    formula: str


def size_constants(lines: list[str]) -> list[SizeConstant]:
    return [SizeConstant(i + 1, m["name"], int(m["value"]), m["formula"])
            for i, line in enumerate(lines) if (m := DECLARATION.match(line))]


def _code(line: str) -> str:
    """The code of a line; the Noir `// @outputs:` line counts as code (its
    counts are sizes, written with the constants' names)."""
    if line.strip().startswith("// @outputs:"):
        return line.split(":", 1)[1]
    return line.split("//", 1)[0]


def check(path: Path, fixture: dict) -> tuple[list[str], list[str]]:
    """(errors, warnings) of a baseline file's size constants."""
    lines = path.read_text().splitlines()
    dims = dimensions(fixture)
    errors, warnings = [], []
    constants = size_constants(lines)
    for c in constants:
        try:
            want = evaluate(c.formula, dims)
        except (ValueError, SyntaxError) as e:
            errors.append(f"line {c.line}: {c.name}: formula {c.formula!r}: {e}")
            continue
        if want != c.value:
            errors.append(f"line {c.line}: {c.name} = {c.value}, but {c.formula} = {want} at the fixture's shapes")
    first_fn = next((i + 1 for i, l in enumerate(lines) if re.match(r"\s*(pub\s+)?(unconstrained\s+)?fn\s", l)), None)
    for c in constants:
        if first_fn is not None and c.line > first_fn:
            errors.append(f"line {c.line}: size constant {c.name} after the first function (declare it at the top)")
    declared = {c.line for c in constants}
    suspicious = size_like(dims, fixture)
    for i, line in enumerate(lines):
        if i + 1 in declared:
            continue
        if INDEX_WIDTH.search(_code(line)):
            warnings.append(f"line {i + 1}: literal index width (it depends on the array length: "
                            f"make it a size constant): `{line.strip()}`")
        hits = sorted({int(n) for n in INTEGER.findall(_code(line)) if int(n) in suspicious})
        if hits:
            warnings.append(f"line {i + 1}: size-like number {', '.join(map(str, hits))} outside the "
                            f"size constants: `{line.strip()}`")
    return errors, warnings


def instantiate(text: str, fixture: dict) -> str:
    """The baseline at another fixture's shapes: each size constant's value
    replaced by its formula's value there."""
    dims = dimensions(fixture)
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        m = DECLARATION.match(body)
        if m:
            body = f"{m['head']}{evaluate(m['formula'], dims)}{m['tail']}"
        out.append(body + line[len(line.rstrip("\n")):])
    return "".join(out)


def main() -> int:
    from .entries import resolve

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("--file", default="target.rs", choices=["target.rs", "main.nr"])
    ap.add_argument("--fixture", type=Path, help="print the baseline instantiated at this fixture's shapes")
    a = ap.parse_args()
    entry = resolve(a.case)
    path = entry.file(a.file)
    if a.fixture:
        sys.stdout.write(instantiate(path.read_text(), json.loads(a.fixture.read_text())))
        return 0
    errors, warnings = check(path, json.loads(entry.fixture.read_text()))
    for c in size_constants(path.read_text().splitlines()):
        print(f"{c.name} = {c.value}  (= {c.formula})")
    for w in warnings:
        print(f"warn  {w}")
    for e in errors:
        print(f"ERROR {e}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
