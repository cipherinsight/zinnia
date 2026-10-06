"""Inline module-level literal constants into a decorated function's source.

The compiler sees only the source ``inspect.getsource`` returns for the
decorated function, not its module. A kernel that defines ``BET_M = 0.5`` at
module level and reads ``BET_M`` in the circuit would otherwise fail with
``Variable BET_M not found``. The decorators therefore replace every read of
such a name by the literal, unless the function binds the name itself (a
local binding shadows the module-level one, as in Python).

Only names whose current value is an ``int``, ``float``, ``bool`` or ``str``
are inlined. Derived constants are inlined by value too, since the value is
read from the module's globals when the decorator runs.
"""
from __future__ import annotations

import ast
from typing import Any, Dict

from zinnia.compile.lowering.scopes import parameter_names
from zinnia.compile.lowering.syntax import dedent_source


def module_constants(method) -> Dict[str, Any]:
    """The literal constants visible from ``method.__globals__``: every
    non-dunder name whose value is an ``int``, ``float``, ``bool`` or ``str``."""
    module_globals = getattr(method, "__globals__", None)
    if not module_globals:
        return {}
    return {
        name: value
        for name, value in module_globals.items()
        if isinstance(name, str)
        and not name.startswith("__")
        and isinstance(value, (bool, int, float, str))
    }


def _names_bound_anywhere(function: ast.FunctionDef) -> set[str]:
    """Parameters and every name stored or deleted anywhere in ``function``,
    including inside nested functions and comprehensions (a conservative
    over-approximation: a name bound anywhere is never inlined). Every name
    in a loop target counts, also the base of a subscript target
    (``for a[i] in ...``), so the target never becomes a literal."""
    bound = parameter_names(function.args)
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            bound.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
    return bound


class _InlineConstants(ast.NodeTransformer):
    def __init__(self, constants: Dict[str, Any], local_names: set[str]):
        super().__init__()
        self.constants = constants
        self.local_names = local_names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            not isinstance(node.ctx, ast.Load)
            or node.id in self.local_names
            or node.id not in self.constants
        ):
            return node
        return ast.copy_location(ast.Constant(value=self.constants[node.id]), node)


def inline_module_constants(source: str, constants: Dict[str, Any]) -> str:
    """Replace reads of ``constants`` in ``source`` (one function definition)
    by their literal values. Returns ``source`` unchanged when there is
    nothing to inline or it is not a single parseable function."""
    if not constants:
        return source
    try:
        tree = ast.parse(dedent_source(source))
    except SyntaxError:
        return source
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return source
    function = tree.body[0]
    inlined = _InlineConstants(constants, _names_bound_anywhere(function)).visit(function)
    return ast.unparse(ast.fix_missing_locations(inlined))
