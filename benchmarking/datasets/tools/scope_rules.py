"""The scope rules shared by every dataset (DS-1000, NPBench, Pythran).

`datasets/FILTERING.md` states these rules in prose; this module is their one
implementation. A program is out of scope when at least one rule fires. Every
rule looks at program text (and, for DS-1000, at the test inputs; see
`ds1000_scope.py`). No rule looks at whether Zinnia compiles the program:
programs Zinnia cannot compile stay in scope and count as rejections.

Rule `imports` is the author's rule (2026-10-06): a program that imports any
module other than `numpy` (with its submodules) and `math` is out of scope.
The other rules come from the earlier FILTERING.md files, made identical for
the three datasets.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

RULES: dict[str, str] = {
    "imports": "imports or uses a module other than numpy (including its submodules) and math",
    "exception_control_flow": "uses try/except/finally",
    "external_io": "reads or writes files, the network, stdin/stdout, or uses NumPy file I/O",
    "nondeterminism": "draws random numbers or reads the time, the process or the environment",
    "non_numeric_data": "works on strings, bytes, objects, ragged sequences, datetimes or record arrays",
    "non_finite_values": "uses NaN or infinity (literals, isnan/isinf/isfinite), or receives them as input",
    "opaque_runtime": "uses eval/exec/compile/getattr/setattr/globals/locals or import machinery",
    "plotting": "draws plots or uses interactive display",
    "no_callable_kernel": "has no top-level function to call as the kernel",
}

ALLOWED_MODULES = {"numpy", "math"}
# `from __future__ import ...` is a compiler directive, not a module the program uses.
IGNORED_IMPORTS = {"__future__"}

EXTERNAL_IO_CALLS = {
    "open", "input", "print",
    "np.load", "np.save", "np.savez", "np.loadtxt", "np.savetxt", "np.genfromtxt", "np.fromfile",
    "np.memmap", "np.DataSource", "np.tofile",
}
NONDETERMINISTIC_PREFIXES = ("np.random", "numpy.random", "random", "time", "datetime", "uuid", "os.environ")
OPAQUE_CALLS = {"eval", "exec", "compile", "__import__", "getattr", "setattr", "globals", "locals"}
NON_NUMERIC_NUMPY_NAMES = {
    "object", "object_", "str_", "string_", "bytes_", "unicode_", "datetime64", "timedelta64",
    "recarray", "record", "char",
}
STRING_METHODS = {"replace", "split", "strip", "lstrip", "rstrip", "lower", "upper", "startswith",
                  "endswith", "join", "encode", "decode", "format"}
NON_FINITE_NAMES = {"nan", "NaN", "NAN", "inf", "Inf", "infty", "Infinity", "PINF", "NINF",
                    "isnan", "isinf", "isfinite", "isposinf", "isneginf", "nan_to_num",
                    "nanmax", "nanmin", "nansum", "nanmean", "nanstd", "nanvar", "nanmedian",
                    "nanpercentile", "nanargmax", "nanargmin", "nanprod", "nancumsum"}
NON_FINITE_FLOAT_STRINGS = {"nan", "inf", "-inf", "infinity", "-infinity"}


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _numpy_name(name: str) -> str | None:
    """`np.inf` -> `inf`; `numpy.linalg.norm` -> `linalg.norm`; else None."""
    for prefix in ("np.", "numpy."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


@dataclass
class RuleHits:
    rules: dict[str, list[str]] = field(default_factory=dict)   # rule -> evidence

    def add(self, rule: str, evidence: str) -> None:
        assert rule in RULES, rule
        items = self.rules.setdefault(rule, [])
        if evidence not in items:
            items.append(evidence)

    def merge(self, other: "RuleHits") -> None:
        for rule, items in other.rules.items():
            for item in items:
                self.add(rule, item)

    def sorted(self) -> dict[str, list[str]]:
        return {rule: self.rules[rule] for rule in RULES if rule in self.rules}


class ScopeVisitor(ast.NodeVisitor):
    """Applies the text rules to one program's AST."""

    def __init__(self) -> None:
        self.hits = RuleHits()

    def _import(self, module: str) -> None:
        root = module.split(".")[0]
        if root not in ALLOWED_MODULES and root not in IGNORED_IMPORTS:
            self.hits.add("imports", f"import {module}")
        if root in {"random", "time", "datetime", "uuid", "secrets"}:
            self.hits.add("nondeterminism", f"import {module}")
        if root in {"importlib"}:
            self.hits.add("opaque_runtime", f"import {module}")
        if root in {"matplotlib", "pylab"}:
            self.hits.add("plotting", f"import {module}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._import(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self._import(node.module)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.hits.add("exception_control_flow", "try statement")
        self.generic_visit(node)

    visit_TryStar = visit_Try  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        if name:
            canonical = name.replace("numpy.", "np.", 1) if name.startswith("numpy.") else name
            if canonical in EXTERNAL_IO_CALLS or canonical.endswith(".tofile"):
                self.hits.add("external_io", f"{name}()")
            if canonical in OPAQUE_CALLS or canonical.startswith("importlib."):
                self.hits.add("opaque_runtime", f"{name}()")
            if any(canonical == p or canonical.startswith(p + ".") for p in NONDETERMINISTIC_PREFIXES):
                self.hits.add("nondeterminism", f"{name}()")
            if canonical.startswith(("plt.", "matplotlib.", "pylab.")):
                self.hits.add("plotting", f"{name}()")
            if (canonical.split(".")[-1] in STRING_METHODS and isinstance(node.func, ast.Attribute)
                    and _numpy_name(canonical) is None):
                self.hits.add("non_numeric_data", f".{canonical.split('.')[-1]}() string method")
            if canonical in {"float", "np.float64", "np.float32"} and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) \
                        and arg.value.strip().lower() in NON_FINITE_FLOAT_STRINGS:
                    self.hits.add("non_finite_values", f"{name}({arg.value!r})")
            if canonical in {"str", "chr", "ord", "repr", "bytes", "bytearray"}:
                self.hits.add("non_numeric_data", f"{name}()")
        for keyword in node.keywords:
            if keyword.arg == "dtype":
                self._dtype(keyword.value)
        self.generic_visit(node)

    def _dtype(self, value: ast.expr) -> None:
        name = dotted_name(value)
        if name and (name.split(".")[-1] in NON_NUMERIC_NUMPY_NAMES or name in {"str", "object", "bytes"}):
            self.hits.add("non_numeric_data", f"dtype={name}")
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if value.value.lstrip("<>=|").startswith(("U", "S", "O", "M", "m", "str", "object", "datetime")):
                self.hits.add("non_numeric_data", f"dtype={value.value!r}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = dotted_name(node)
        numpy_name = _numpy_name(name) if name else None
        if numpy_name is not None:
            leaf = numpy_name.split(".")[-1]
            if numpy_name in NON_NUMERIC_NUMPY_NAMES or numpy_name.startswith("char."):
                self.hits.add("non_numeric_data", name)
            if leaf in NON_FINITE_NAMES:
                self.hits.add("non_finite_values", name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # `from numpy import inf, isnan` brings the names in unqualified.
        if node.id in {"inf", "nan", "isnan", "isinf", "isfinite", "NaN", "Inf"}:
            self.hits.add("non_finite_values", node.id)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, float) and (node.value != node.value or node.value in (float("inf"), float("-inf"))):
            self.hits.add("non_finite_values", repr(node.value))
        self.generic_visit(node)


def text_rules(tree: ast.AST) -> RuleHits:
    visitor = ScopeVisitor()
    visitor.visit(tree)
    return visitor.hits
