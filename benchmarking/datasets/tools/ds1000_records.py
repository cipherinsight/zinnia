"""Read DS-1000 NumPy records: test inputs, the exec_context around the solution,
and the names a solution reads.

A DS-1000 record runs its reference solution like this (see `code_context`):

    test_input, expected = generate_test_case(k)        # k = 1 .. test_case_cnt
    exec(exec_context.replace("[insert]", solution), {"test_input": test_input})
    assert exec_test(env["result"], expected)

`exec_context` has a *prelude* (the lines before `[insert]`, which bind the
solution's inputs from `test_input`), sometimes a function header `def f(...):`
that wraps the solution, and a *suffix* (the lines after `[insert]`, usually
`result = <expr>`). The value DS-1000 checks is `result` after the suffix ran.

This module evaluates those pieces without pandas or any other library: test
inputs are produced by running DS-1000's own `define_test_input` with NumPy only.
Two DS-1000 NumPy problems (0398, 0399) wrap a NumPy array in `pd.Series`; a
small stand-in turns `pd.Series(x)` into `np.asarray(x)` and records that it did,
so the caller can report the conversion. Any other pandas use raises.
"""

from __future__ import annotations

import ast
import builtins
import copy
import json
import math
import re
import symtable
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DATASETS = Path(__file__).resolve().parent.parent
MANIFEST = DATASETS / "ds1000_numpy" / "manifest.json"


class RecordError(Exception):
    """A record whose structure this module cannot read mechanically."""


def load_manifest(path: Path = MANIFEST) -> dict[int, dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))["programs"]
    return {row["problem_id"]: row for row in rows}


# ---------------------------------------------------------------------------
# exec_context
# ---------------------------------------------------------------------------


@dataclass
class ExecContext:
    prelude: list[ast.stmt]          # statements before [insert], imports included
    function_header: ast.FunctionDef | None   # `def f(...):` wrapping the solution
    suffix: list[ast.stmt]           # statements after [insert]
    suffix_source: str


def parse_exec_context(code_context: str) -> ExecContext:
    match = re.search(r'exec_context\s*=\s*r?"""(.*?)"""', code_context, re.S)
    if not match:
        raise RecordError("no exec_context in code_context")
    head, sep, tail = match.group(1).partition("[insert]")
    if not sep:
        raise RecordError("exec_context has no [insert] marker")
    head_lines = head.rstrip().splitlines()
    header: ast.FunctionDef | None = None
    if head_lines and re.match(r"\s*def\s+\w+\s*\(.*\)\s*:\s*$", head_lines[-1]):
        header_tree = ast.parse(head_lines[-1].strip() + "\n    pass\n")
        header = header_tree.body[0]  # type: ignore[assignment]
        head_lines = head_lines[:-1]
    prelude = ast.parse(textwrap.dedent("\n".join(head_lines))).body
    suffix_source = textwrap.dedent(tail).strip("\n")
    suffix = ast.parse(suffix_source).body if suffix_source.strip() else []
    return ExecContext(prelude, header, suffix, suffix_source)


def imported_modules(statements: list[ast.stmt]) -> list[str]:
    """Top-level module names imported by `statements` (e.g. `numpy`, `pandas`)."""
    modules: list[str] = []
    for node in statements:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                modules.extend(alias.name for alias in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.module:
                modules.append(sub.module)
    return modules


def is_import(node: ast.stmt) -> bool:
    return isinstance(node, (ast.Import, ast.ImportFrom))


def assigned_names(node: ast.stmt) -> list[str]:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    names: list[str] = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
    return names


def loaded_names(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)}


def reads_test_input(node: ast.stmt) -> bool:
    return "test_input" in loaded_names(node)


# ---------------------------------------------------------------------------
# Test inputs
# ---------------------------------------------------------------------------


class _PandasStandIn:
    """Supports exactly `pd.Series(values)`, which becomes a NumPy array."""

    def __init__(self) -> None:
        self.used: list[str] = []

    def Series(self, data, *args, **kwargs):  # noqa: N802 (pandas name)
        if args or kwargs:
            raise RecordError("pd.Series with extra arguments is not supported")
        self.used.append("Series")
        return np.asarray(data)

    def __getattr__(self, name: str):
        raise RecordError(f"test input needs pandas.{name}")


def _define_test_input_source(code_context: str) -> ast.FunctionDef:
    tree = ast.parse(code_context)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "generate_test_case":
            for inner in node.body:
                if isinstance(inner, ast.FunctionDef) and inner.name == "define_test_input":
                    return inner
    raise RecordError("no generate_test_case.define_test_input in code_context")


@dataclass
class TestCase:
    """The names a solution sees for one DS-1000 test case."""

    case_id: int
    env: dict[str, Any]                 # every name bound by the prelude
    pandas_used: list[str] = field(default_factory=list)


def evaluate_test_case(record: dict[str, Any], case_id: int) -> TestCase:
    """Run DS-1000's `define_test_input(case_id)` and the exec_context prelude.

    NumPy's global RNG is reset to seed 0 first so that prelude lines that draw
    without seeding (unused by every in-scope solution) are reproducible.
    """
    pandas = _PandasStandIn()
    func = _define_test_input_source(record["code_context"])
    namespace: dict[str, Any] = {"np": np, "numpy": np, "pd": pandas, "copy": copy, "math": math}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "<define_test_input>", "exec"), namespace)
    np.random.seed(0)
    try:
        test_input = namespace["define_test_input"](case_id)
    except RecordError:
        raise
    except NameError as exc:
        raise RecordError(f"test input needs a library: {exc}") from exc
    context = parse_exec_context(record["code_context"])
    env: dict[str, Any] = {"np": np, "numpy": np, "pd": pandas, "test_input": test_input}
    for node in context.prelude:
        if is_import(node):
            continue
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<prelude>", "exec"), env)
    for name in ("np", "numpy", "pd", "test_input", "__builtins__"):
        env.pop(name, None)
    return TestCase(case_id, env, list(pandas.used))


def test_case_ids(record: dict[str, Any]) -> list[int]:
    count = record.get("test_case_cnt") or 1
    return list(range(1, int(count) + 1))


# ---------------------------------------------------------------------------
# Program inputs
# ---------------------------------------------------------------------------

BUILTIN_NAMES = set(dir(builtins)) | {"np", "numpy"}


def program_source(record: dict[str, Any], context: ExecContext) -> str:
    """The solution as DS-1000 runs it: the optional `def f(...)` header, the
    reference code, then the suffix. Prelude statements are left out so that
    the names they bind show up as free (input) names."""
    reference = record["reference_code"]
    if context.function_header is not None:
        header = ast.unparse(context.function_header).splitlines()[0]
        body = reference if reference.strip() else "    pass\n"
        return f"{header}\n{body.rstrip()}\n{context.suffix_source}\n"
    return f"{textwrap.dedent(reference).rstrip()}\n{context.suffix_source}\n"


class _ModuleReads(ast.NodeVisitor):
    """Names read at module level before any statement bound them.

    Statements are visited in execution order and every assignment visits its
    right-hand side before its targets, so `a = a ** power` reads `a`. Names a
    comprehension binds are local to it. Function and lambda bodies are not
    visited here; `_function_globals` handles them.
    """

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.free: list[str] = []
        self._local: list[set[str]] = []

    def _read(self, name: str) -> None:
        if any(name in scope for scope in self._local):
            return
        if name not in self.bound and name not in self.free:
            self.free.append(name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._read(node.id)
        elif self._local:
            self._local[-1].add(node.id)
        else:
            self.bound.add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._read(node.target.id)
        self.visit(node.target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self.visit(node.target)
        for stmt in node.body + node.orelse:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for default in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(default)
        self.bound.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in node.args.defaults:
            self.visit(default)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def _comprehension(self, node: ast.AST, elements: list[ast.expr]) -> None:
        generators = node.generators  # type: ignore[attr-defined]
        self.visit(generators[0].iter)
        self._local.append(set())
        for index, gen in enumerate(generators):
            if index:
                self.visit(gen.iter)
            self.visit(gen.target)
            for cond in gen.ifs:
                self.visit(cond)
        for element in elements:
            self.visit(element)
        self._local.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp  # type: ignore[assignment]

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node, [node.key, node.value])


def _function_globals(source: str) -> list[str]:
    """Implicit globals read inside functions and lambdas of `source`."""
    found: list[str] = []

    def visit(table: symtable.SymbolTable) -> None:
        if table.get_type() != "module":
            for sym in table.get_symbols():
                if sym.is_referenced() and sym.is_global() and sym.get_name() not in found:
                    found.append(sym.get_name())
        for child in table.get_children():
            visit(child)

    visit(symtable.symtable(source, "<solution>", "exec"))
    return found


def free_names(source: str) -> list[str]:
    """Names a program reads that it does not bind first, in reading order.

    A name read inside a function counts when no module-level statement binds
    it anywhere (the function may run after the binding statement). Builtin
    names are included: DS-1000 sometimes binds an input named `eval` or `min`.
    """
    visitor = _ModuleReads()
    for stmt in ast.parse(source).body:
        visitor.visit(stmt)
    names = list(visitor.free)
    for name in _function_globals(source):
        if name not in visitor.bound and name not in names:
            names.append(name)
    return names


@dataclass
class ProgramInputs:
    """The parameters of the kernel for one DS-1000 program, in prelude order."""

    names: list[str]
    dead_test_inputs: list[str]       # bound from test_input, never read
    prelude_imports: list[str]       # modules the prelude imports
    imported_reads: list[str]        # names the solution reads that a prelude import binds


def program_inputs(record: dict[str, Any]) -> ProgramInputs:
    """Kernel parameters: every name the solution reads that the prelude binds,
    plus test inputs nobody reads (DS-1000 passes them; the kernel keeps them so
    its signature matches the DS-1000 program). Order follows the prelude."""
    context = parse_exec_context(record["code_context"])
    reads = set(free_names(program_source(record, context)))
    import_bound: set[str] = set()
    for node in context.prelude:
        if is_import(node):
            visitor = _ModuleReads()
            visitor.visit(node)
            import_bound |= visitor.bound
    imported_reads = sorted(reads & import_bound)
    reads -= import_bound
    binding_order: list[str] = []
    test_input_names: list[str] = []
    read_by_prelude: set[str] = set()
    for node in context.prelude:
        if is_import(node):
            continue
        names = assigned_names(node)
        if reads_test_input(node):
            test_input_names.extend(names)
        else:
            read_by_prelude |= loaded_names(node)
        binding_order.extend(n for n in names if n not in binding_order)
    unbound = sorted(reads - set(binding_order) - BUILTIN_NAMES)
    if unbound:
        raise RecordError(f"solution reads names no prelude statement binds: {unbound}")
    dead = [n for n in test_input_names if n not in reads and n not in read_by_prelude]
    names = [n for n in binding_order if n in reads or n in dead]
    return ProgramInputs(names, dead, imported_modules(context.prelude), imported_reads)
