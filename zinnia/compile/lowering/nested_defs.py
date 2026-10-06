"""Lift self-contained local functions of a circuit into chips (decorator stage).

``@zk_circuit`` runs this pass while it still has the decorated function's
module globals. A ``def`` directly in the circuit body becomes a chip under
its own name when every name it reads is

- its own parameter or a name it binds,
- a Python builtin, or
- a global of the circuit's module (for example ``math`` or another chip).

Such a function behaves the same wherever it is called, so compiling it as a
chip keeps Python's meaning; as a chip it may also call itself recursively.
A local function that reads any of the circuit's own variables (a closure),
or a name that is bound nowhere, stays in the body: the compile-time helper
lifter (`helpers.py`) lifts it with the variables it reads passed as extra
arguments, or rejects it. A ``def`` nested inside a branch or loop is never
lifted here.

The free-name test here is deliberately coarser than `scopes.free_names`:
names bound only as comprehension targets count as free, so a function that
uses a comprehension variable not bound elsewhere is left to the helper
lifter.
"""
from __future__ import annotations

import ast
import builtins
from typing import Any, Dict, List, Tuple

from zinnia.compile.lowering.scopes import parameter_names
from zinnia.compile.lowering.syntax import result_assignments_to_returns

_BUILTIN_NAMES = frozenset(dir(builtins))


class _AssignedNames(ast.NodeVisitor):
    """Names a function body binds by assignment, loop or ``with`` targets,
    or nested ``def`` (whose own body is not entered)."""

    def __init__(self, bound: set[str]):
        self.bound = bound

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._add_target(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._add_target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.target is not None:
            self._add_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._add_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._add_target(item.optional_vars)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)

    def _add_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.bound.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._add_target(element)
        # A subscript or attribute target writes into an existing object.


class _UnboundReads(ast.NodeVisitor):
    """Names read (outside nested ``def`` bodies) that are neither bound nor builtins."""

    def __init__(self, bound: set[str]):
        self.bound = bound
        self.free: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id not in self.bound and node.id not in _BUILTIN_NAMES:
            self.free.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass


def _external_reads(function: ast.FunctionDef) -> set[str]:
    """The non-builtin names ``function`` reads but does not bind."""
    bound = parameter_names(function.args)
    assigned = _AssignedNames(bound)
    for statement in function.body:
        assigned.visit(statement)
    reads = _UnboundReads(bound)
    for statement in function.body:
        reads.visit(statement)
    return reads.free


def _circuit_locals(circuit: ast.FunctionDef) -> set[str]:
    """Names the circuit binds that a local function could close over: its
    parameters, simple assignments at the top level of its body, and its
    local function names."""
    names = {a.arg for a in circuit.args.posonlyargs + circuit.args.args + circuit.args.kwonlyargs}
    for statement in circuit.body:
        if isinstance(statement, ast.Assign):
            names.update(t.id for t in statement.targets if isinstance(t, ast.Name))
        elif isinstance(statement, ast.FunctionDef):
            names.add(statement.name)
    return names


def _as_chip_source(function: ast.FunctionDef) -> str:
    chip = ast.FunctionDef(
        name=function.name,
        args=function.args,
        body=result_assignments_to_returns(function.body),
        decorator_list=[],
        returns=function.returns,
        type_comment=getattr(function, "type_comment", None),
    )
    ast.copy_location(chip, function)
    return ast.unparse(chip)


def lift_self_contained_defs(
    source: str, module_globals: Dict[str, Any]
) -> Tuple[str, List[Tuple[str, str]]]:
    """Lift the circuit's self-contained local functions into chips.

    ``source`` holds the circuit's definition (its first top-level ``def``).
    Returns ``(new_source, [(chip_name, chip_source), ...])``; the source is
    unchanged (not re-printed) when nothing is lifted.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    circuit = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
    if circuit is None:
        return source, []

    module_names = set(module_globals) if module_globals else set()
    circuit_locals = _circuit_locals(circuit)
    kept: List[ast.stmt] = []
    lifted: List[Tuple[str, str]] = []
    for statement in circuit.body:
        if not isinstance(statement, ast.FunctionDef):
            kept.append(statement)
            continue
        reads = _external_reads(statement)
        closes_over_circuit = reads & circuit_locals
        unbound = reads - circuit_locals - module_names
        if closes_over_circuit or unbound:
            kept.append(statement)
            continue
        lifted.append((statement.name, _as_chip_source(statement)))

    if not lifted:
        return source, []
    circuit.body = kept if kept else [ast.Pass()]
    return ast.unparse(tree), lifted
