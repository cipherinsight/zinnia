"""Helpers for resolving module-level literal constants referenced inside
@zk_circuit / @zk_chip bodies.

Background
----------
The IR generator only sees the source captured by ``inspect.getsource`` for
the decorated function — it has no visibility into the surrounding module
scope. As a result, a benchmark that defines ``BET_M = 0.5`` at module level
and references ``BET_M`` inside the circuit body fails with
``Variable BET_M not found``.

This module provides two utilities used by the @zk_circuit and @zk_chip
decorators:

* ``extract_module_constants(method)`` — walks ``method.__globals__`` and
  returns a ``dict[str, Any]`` of names whose values are statically a single
  Python literal (int, float, bool, str). Anything else (functions, classes,
  derived computations, NDArray instances, etc.) is excluded.
* ``substitute_module_constants(source, constants)`` — re-parses the function
  source, replaces ``ast.Name`` references (in Load ctx) to known constants
  with ``ast.Constant`` literals (skipping names shadowed by locals), and
  returns the unparsed source.

Limitation: derived constants (``C = 16 * A * B``) are intentionally NOT
supported — see the kanban ticket ``compiler.module-const-and-chip-default-args``.
"""
from __future__ import annotations

import ast
from typing import Any, Dict


_LITERAL_TYPES = (int, float, bool, str)


def extract_module_constants(method) -> Dict[str, Any]:
    """Return literal constants visible from ``method.__globals__``.

    Only int / float / bool / str values whose name is a valid Python
    identifier and not dunder are returned. ``True``/``False``/``None``
    builtins are also filtered out (they always resolve at the language
    level).
    """
    globs = getattr(method, "__globals__", None)
    if not globs:
        return {}
    out: Dict[str, Any] = {}
    for name, value in globs.items():
        if not isinstance(name, str):
            continue
        if name.startswith("__"):
            continue
        # Note: ``isinstance(True, int)`` is True, so bool branch first.
        if isinstance(value, bool):
            out[name] = value
        elif isinstance(value, (int, float, str)):
            out[name] = value
    return out


class _NameToConstantTransformer(ast.NodeTransformer):
    """Replace ``ast.Name(id=<known>)`` in Load ctx with ``ast.Constant``.

    Names that are bound locally inside the function (parameters, assignment
    targets, for-loop variables, comprehension variables) are NOT replaced —
    a local binding always shadows the module-level constant.
    """

    def __init__(self, constants: Dict[str, Any], local_names: set):
        super().__init__()
        self._constants = constants
        self._locals = local_names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Only replace bare Loads of names that are not locally bound.
        if not isinstance(node.ctx, ast.Load):
            return node
        if node.id in self._locals:
            return node
        if node.id not in self._constants:
            return node
        value = self._constants[node.id]
        new_node = ast.Constant(value=value)
        ast.copy_location(new_node, node)
        return new_node


def _collect_local_names(func_node: ast.FunctionDef) -> set:
    """Names bound inside ``func_node`` — args, assigns, for-loop vars, etc."""
    locals_: set = set()
    # Parameters (positional, kw-only, vararg, kwarg).
    args = func_node.args
    for a in list(args.args) + list(args.kwonlyargs) + list(args.posonlyargs):
        locals_.add(a.arg)
    if args.vararg is not None:
        locals_.add(args.vararg.arg)
    if args.kwarg is not None:
        locals_.add(args.kwarg.arg)

    # Walk the body for store-context names.
    for node in ast.walk(func_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            locals_.add(node.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    locals_.add(sub.id)
        elif isinstance(node, (ast.comprehension,)):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    locals_.add(sub.id)
    return locals_


def substitute_module_constants(source: str, constants: Dict[str, Any]) -> str:
    """Parse ``source`` (a single function def), substitute references to
    ``constants`` with literal AST constants, and return the new source.

    If ``constants`` is empty, returns ``source`` unchanged.
    """
    if not constants:
        return source

    # Match the indentation handling done by ZinniaCompiler.fix_source_indentation
    # so that ast.parse succeeds on dedented chip / circuit sources.
    lines = source.split("\n")
    min_indent = None
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if min_indent is None or indent < min_indent:
                min_indent = indent
    if min_indent is None:
        min_indent = 0
    fixed = "\n".join(line[min_indent:] for line in lines)

    try:
        tree = ast.parse(fixed)
    except SyntaxError:
        return source

    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return source

    func = tree.body[0]
    local_names = _collect_local_names(func)
    transformer = _NameToConstantTransformer(constants, local_names)
    new_func = transformer.visit(func)
    ast.fix_missing_locations(new_func)
    try:
        return ast.unparse(new_func)
    except AttributeError:
        # ast.unparse is only available on Python 3.9+. Zinnia targets 3.10.
        return source
