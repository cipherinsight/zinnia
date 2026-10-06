"""Imports: which modules a kernel may import, and how imported names are spelled.

The rule. A kernel that imports any module other than ``numpy`` and ``math``
(or their submodules, such as ``numpy.linalg``) is outside Zinnia's scope and
is rejected with ``out of scope: imports <module>``. ``ALLOWED_IMPORT_ROOTS``
is the only definition of the allowed set. The rule is checked

- on import statements in a circuit's source (`resolve_import_aliases`),
- on import statements in a chip body (the transformer, `check_import_statement`),
- on module-level imports whose names a decorated function uses
  (`check_module_level_imports`, called by the decorators, because
  ``inspect.getsource`` gives the compiler only the function itself).

The spelling. The compiler knows NumPy as ``np`` and math as ``math``.
`resolve_import_aliases` replaces every name an allowed import binds by that
spelling (``import numpy as xp`` → ``xp.sum`` becomes ``np.sum``;
``from numpy.linalg import norm`` → ``norm`` becomes ``np.linalg.norm``) and
removes the import statements, following Python's scoping: a local binding of
the same name shadows the import. The decorators see only a function's own
source, so `resolve_module_level_imports` applies the same spelling to the
names the function reads from its module's top-level imports
(``import numpy`` → ``numpy.exp`` becomes ``np.exp``).
"""
from __future__ import annotations

import ast
import copy
from pathlib import Path

from zinnia.compile.lowering.scopes import bound_names
from zinnia.compile.lowering.syntax import FunctionLoweringError, dedent_source, load
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


def imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    """The module names an import statement imports."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return ["." * node.level + (node.module or "")]


def check_import_statement(node: ast.Import | ast.ImportFrom) -> None:
    for module in imported_modules(node):
        check_import_in_scope(module)


def check_module_level_imports(method, source: str) -> None:
    """Apply the import rule to module-level imports a function uses.

    Reads the defining module's top-level imports and rejects the function if
    any name it loads was bound by an import of a module outside the allowed
    set. A function whose module has no file (e.g. defined in a notebook) is
    not checked here.
    """
    module_file = (getattr(method, "__globals__", None) or {}).get("__file__")
    if not module_file:
        return
    try:
        module_tree = ast.parse(Path(module_file).read_text())
        function_tree = ast.parse(dedent_source(source))
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


def _module_expression(module: str) -> ast.expr:
    """The circuit spelling of an allowed module: NumPy is always ``np``;
    ``math`` keeps its name; submodules are attribute chains."""
    root, *parts = module.split(".")
    value: ast.expr = load("np" if root == "numpy" else root)
    for part in parts:
        value = ast.Attribute(value=value, attr=part, ctx=ast.Load())
    return value


def import_bindings(node: ast.Import | ast.ImportFrom) -> dict[str, ast.expr]:
    """The names an allowed import binds, each mapped to its circuit spelling.

    Imports of other modules are rejected (the rule above).
    """
    check_import_statement(node)
    result = {}
    if isinstance(node, ast.Import):
        for alias in node.names:
            # `import numpy.linalg` binds `numpy`; with `as`, the alias
            # names the submodule itself.
            module = alias.name if alias.asname else alias.name.split(".")[0]
            result[alias.asname or module] = _module_expression(module)
    else:
        for alias in node.names:
            if alias.name == "*":
                raise FunctionLoweringError("Wildcard imports require explicit member names")
            result[alias.asname or alias.name] = ast.Attribute(
                value=_module_expression(node.module), attr=alias.name, ctx=ast.Load())
    return result


def _parameters(node: ast.Lambda) -> list[ast.arg]:
    return node.args.posonlyargs + node.args.args + node.args.kwonlyargs


class _AliasResolver(ast.NodeTransformer):
    """Replaces names bound by imports with their circuit spelling.

    ``aliases`` holds the import bindings visible in the current scope.
    """

    def __init__(self, aliases: dict[str, ast.expr] | None = None):
        self.aliases = dict(aliases or {})

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.aliases:
            return ast.copy_location(copy.deepcopy(self.aliases[node.id]), node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Defaults and decorators are evaluated in the enclosing scope.
        node.decorator_list = [self.visit(x) for x in node.decorator_list]
        node.args.defaults = [self.visit(x) for x in node.args.defaults]
        node.args.kw_defaults = [self.visit(x) if x else None for x in node.args.kw_defaults]
        inherited = dict(self.aliases)
        for name in bound_names(node):
            inherited.pop(name, None)
        imported = {}
        for statement in node.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                imported.update(import_bindings(statement))
        # An import binds its names in the whole function, even on paths
        # before the import runs.
        for name in imported:
            inherited.pop(name, None)
        assigned = bound_names(node)
        if set(imported) & assigned:
            raise FunctionLoweringError('A NumPy import cannot share a local binding with a parameter or assignment')
        body_resolver = _AliasResolver(inherited)
        body = []
        for statement in node.body:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                body_resolver.aliases.update(import_bindings(statement))
                continue
            # Reject references before a local import rather than resolving to
            # an inherited module binding that Python would regard as unbound.
            pending = set(imported) - set(body_resolver.aliases)
            if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in pending
                   for n in ast.walk(statement)):
                raise FunctionLoweringError('Local NumPy imports must precede their uses and nested helper definitions')
            body.append(body_resolver.visit(statement))
        reads_np = any(isinstance(n, ast.Name) and n.id == 'np' and isinstance(n.ctx, ast.Load)
                       for item in body for n in ast.walk(item))
        if 'np' in assigned and reads_np and (imported or inherited):
            raise FunctionLoweringError('The canonical NumPy name np is shadowed; rename that local binding')
        node.body = body
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        if self.aliases and any(arg.arg == 'np' for arg in _parameters(node)):
            raise FunctionLoweringError('A lambda parameter shadows the canonical NumPy name np')
        aliases = dict(self.aliases)
        for arg in _parameters(node):
            aliases.pop(arg.arg, None)
        node.body = _AliasResolver(aliases).visit(node.body)
        return node

    def _comprehension(self, node: ast.ListComp | ast.GeneratorExp | ast.SetComp | ast.DictComp) -> ast.AST:
        resolver = _AliasResolver(self.aliases)
        for generator in node.generators:
            generator.iter = resolver.visit(generator.iter)
            for target in ast.walk(generator.target):
                if isinstance(target, ast.Name):
                    if target.id == 'np' and resolver.aliases:
                        raise FunctionLoweringError('A comprehension target shadows the canonical NumPy name np')
                    resolver.aliases.pop(target.id, None)
            generator.ifs = [resolver.visit(condition) for condition in generator.ifs]
        if isinstance(node, ast.DictComp):
            node.key = resolver.visit(node.key)
            node.value = resolver.visit(node.value)
        else:
            node.elt = resolver.visit(node.elt)
        return node

    visit_ListComp = _comprehension
    visit_GeneratorExp = _comprehension
    visit_SetComp = _comprehension
    visit_DictComp = _comprehension

    def visit_Import(self, node: ast.Import | ast.ImportFrom) -> ast.AST:
        check_import_statement(node)
        raise FunctionLoweringError('An import must occur directly in a module or function body')

    visit_ImportFrom = visit_Import


def _drop_rebound_names(aliases: dict[str, ast.expr], body: list[ast.stmt]) -> None:
    """Module bindings are resolved when functions execute, after module
    setup, so a later module-level definition of an imported name wins."""
    for statement in body:
        if isinstance(statement, ast.FunctionDef):
            aliases.pop(statement.name, None)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        aliases.pop(name.id, None)


def resolve_import_aliases(module: ast.Module) -> ast.Module:
    """Resolve the names bound by imports in a parsed source file.

    Module-level imports are removed and their names resolved in every
    function that does not rebind them; a function's own imports are resolved
    within that function. Imports anywhere else (inside a loop, a branch)
    are rejected.
    """
    aliases = {}
    body = []
    for statement in module.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            aliases.update(import_bindings(statement))
        else:
            body.append(statement)
    _drop_rebound_names(aliases, body)
    resolver = _AliasResolver(aliases)
    module.body = [resolver.visit(statement) for statement in body]
    return module


def _allowed_bindings(statement: ast.Import | ast.ImportFrom) -> dict[str, ast.expr]:
    """`import_bindings` of a top-level import of allowed modules only. Other
    imports bind nothing here: `check_module_level_imports` rejects a function
    that uses them, and a wildcard import cannot be spelled name by name."""
    allowed = all(module.split(".")[0] in ALLOWED_IMPORT_ROOTS for module in imported_modules(statement))
    wildcard = isinstance(statement, ast.ImportFrom) and any(a.name == "*" for a in statement.names)
    return import_bindings(statement) if allowed and not wildcard else {}


def resolve_module_level_imports(method, source: str) -> str:
    """Spell the names ``source`` (the decorated function ``method``) reads
    from its module's top-level ``numpy``/``math`` imports as the compiler
    does: ``import numpy`` → ``numpy.exp`` becomes ``np.exp``, ``from math
    import sqrt`` → ``sqrt`` becomes ``math.sqrt``. Local bindings shadow the
    imports, as in `resolve_import_aliases`. Returns ``source`` unchanged when
    no name it reads needs another spelling (``import numpy as np``)."""
    module_file = (getattr(method, "__globals__", None) or {}).get("__file__")
    if not module_file:
        return source
    try:
        module_tree = ast.parse(Path(module_file).read_text())
        function_tree = ast.parse(dedent_source(source))
    except (OSError, SyntaxError, ValueError):
        return source
    aliases: dict[str, ast.expr] = {}
    body = []
    for statement in module_tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            aliases.update(_allowed_bindings(statement))
        else:
            body.append(statement)
    _drop_rebound_names(aliases, body)
    respelled = {name: spelling for name, spelling in aliases.items() if ast.unparse(spelling) != name}
    reads = {n.id for n in ast.walk(function_tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if not respelled.keys() & reads:
        return source
    resolved = _AliasResolver(respelled).visit(function_tree)
    return ast.unparse(ast.fix_missing_locations(resolved))
