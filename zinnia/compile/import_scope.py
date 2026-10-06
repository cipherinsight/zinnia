"""Which modules a circuit may import.

A kernel that imports any module other than these (or their submodules,
such as ``numpy.linalg``) is outside Zinnia's scope. Compilation rejects it
with an error that says ``out of scope: imports <module>``. This is the only
place the allowed set is defined.
"""
from __future__ import annotations

import ast
from pathlib import Path

from zinnia.debug.exception.base import ZinniaException

ALLOWED_IMPORT_ROOTS = frozenset({"numpy", "math"})

# Zinnia's own API (type annotations, decorators, precondition predicates) is
# how a kernel is written, not a dependency of the computation.
FRAMEWORK_ROOT = "zinnia"


class OutOfScopeImportError(ZinniaException, ValueError):
    """The program imports a module outside ``ALLOWED_IMPORT_ROOTS``. It is a
    ``ZinniaException`` so that it surfaces unchanged when raised while a
    decorator reads the program (``@zk_circuit`` checks module-level imports
    when it is applied)."""


def check_import_in_scope(module: str) -> None:
    """Raise :class:`OutOfScopeImportError` unless ``module`` is allowed.

    ``module`` is a dotted absolute module name; a relative import is passed
    with its leading dots and is always out of scope.
    """
    root = module.split(".")[0]
    if not root or root not in ALLOWED_IMPORT_ROOTS:
        raise OutOfScopeImportError(f"out of scope: imports {module}")


def imported_module(node: ast.Import | ast.ImportFrom) -> list[str]:
    """The module names an import statement imports."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return ["." * node.level + (node.module or "")]


def check_import_statement(node: ast.Import | ast.ImportFrom) -> None:
    for module in imported_module(node):
        check_import_in_scope(module)


def check_module_level_imports(method, source: str) -> None:
    """Apply the import rule to module-level imports a function uses.

    ``inspect.getsource`` gives the compiler only the function, so imports at
    the top of its module are invisible to the transformer. This reads the
    defining module's top-level imports and rejects the function if any name
    it loads was bound by an import of a module outside the allowed set.
    """
    module_file = (getattr(method, "__globals__", None) or {}).get("__file__")
    if not module_file:
        return
    try:
        module_tree = ast.parse(Path(module_file).read_text())
        function_tree = ast.parse(_dedent(source))
    except (OSError, SyntaxError, ValueError):
        return
    bindings: dict[str, str] = {}
    for statement in module_tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(statement, ast.ImportFrom):
            module = "." * statement.level + (statement.module or "")
            for alias in statement.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = module
    used = {n.id for n in ast.walk(function_tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    for name in sorted(used & set(bindings)):
        if bindings[name].split(".")[0] != FRAMEWORK_ROOT:
            check_import_in_scope(bindings[name])


def _dedent(source: str) -> str:
    lines = source.split("\n")
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    cut = min(indents) if indents else 0
    return "\n".join(line[cut:] for line in lines)
