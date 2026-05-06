"""Auto-lift pure nested ``def`` declarations inside @zk_circuit bodies to
module-level @zk_chip helpers.

A nested ``def`` is *pure* (and therefore safe to lift) if every name it
references is bound to one of:
- its own parameters,
- a name assigned inside its own body,
- a Python builtin (``range``, ``len``, ``abs``, ...),
- a module-global value visible to the outer ``@zk_circuit`` function.

If a nested def captures any name from the *outer function's locals* —
parameters or names assigned in the outer body before / around the def —
it is a closure and cannot be safely lifted. Those are left in place and
will be rejected by the transformer with the existing diagnostic.

Top-level conditional defs (e.g., ``if cond: def f(): ...``) are not
lifted — their existence depends on runtime control flow.
"""
from __future__ import annotations

import ast
import builtins as _builtins
from typing import Any, Dict, List, Tuple


_BUILTIN_NAMES = frozenset(dir(_builtins))


def _free_names(node: ast.FunctionDef) -> set:
    """Return names referenced in ``node`` that are NOT bound by its own
    parameters or local assignments. Builtins are excluded.
    """
    bound: set = set(a.arg for a in node.args.args)
    bound.update(a.arg for a in node.args.kwonlyargs)
    bound.update(a.arg for a in node.args.posonlyargs)
    if node.args.vararg:
        bound.add(node.args.vararg.arg)
    if node.args.kwarg:
        bound.add(node.args.kwarg.arg)

    # Walk to collect locally assigned names.
    class _AssignVisitor(ast.NodeVisitor):
        def visit_Assign(self, n):
            for tgt in n.targets:
                self._capture_targets(tgt)
            self.generic_visit(n)

        def visit_AugAssign(self, n):
            self._capture_targets(n.target)
            self.generic_visit(n)

        def visit_AnnAssign(self, n):
            if n.target is not None:
                self._capture_targets(n.target)
            self.generic_visit(n)

        def visit_For(self, n):
            self._capture_targets(n.target)
            self.generic_visit(n)

        def visit_With(self, n):
            for item in n.items:
                if item.optional_vars is not None:
                    self._capture_targets(item.optional_vars)
            self.generic_visit(n)

        def visit_FunctionDef(self, n):
            bound.add(n.name)
            # Don't recurse into inner scopes — their locals are theirs.

        def _capture_targets(self, tgt):
            if isinstance(tgt, ast.Name):
                bound.add(tgt.id)
            elif isinstance(tgt, (ast.Tuple, ast.List)):
                for elt in tgt.elts:
                    self._capture_targets(elt)
            # Subscripts/Attributes target an existing name; ignore.

    visitor = _AssignVisitor()
    for stmt in node.body:
        visitor.visit(stmt)

    free: set = set()
    class _NameVisitor(ast.NodeVisitor):
        def visit_Name(self, n):
            if isinstance(n.ctx, ast.Load) and n.id not in bound and n.id not in _BUILTIN_NAMES:
                free.add(n.id)
        def visit_FunctionDef(self, n):
            # Skip inner function bodies; their captures aren't this def's.
            pass

    nv = _NameVisitor()
    for stmt in node.body:
        nv.visit(stmt)
    return free


def _rewrite_zinnia_result_to_return(body: List[ast.stmt]) -> List[ast.stmt]:
    """Replace ``_zinnia_result = X`` with ``return X`` so a circuit-style
    inner def can be parsed as a chip."""
    out: List[ast.stmt] = []
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "_zinnia_result"
        ):
            ret = ast.Return(value=stmt.value)
            ast.copy_location(ret, stmt)
            out.append(ret)
        else:
            out.append(stmt)
    return out


def autolift_nested_defs(source: str, module_globals: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]]]:
    """Find pure inner ``def`` nodes in the @zk_circuit body and extract
    them. Returns ``(new_source, [(chip_name, chip_source), ...])``.

    Inner defs that capture outer-function locals are left in the body —
    the transformer will surface a clear diagnostic for them.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []

    # Find the outermost decorated function. The decorator stack has been
    # peeled off by the time this is called from @zk_circuit, but
    # @zk_circuit is itself the entry — we just take the first FunctionDef.
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn = node
            break
    if fn is None:
        return source, []

    module_names = set(module_globals.keys()) if module_globals else set()
    outer_param_names = set(a.arg for a in fn.args.args) | \
        set(a.arg for a in fn.args.kwonlyargs) | \
        set(a.arg for a in fn.args.posonlyargs)

    # Names assigned at the top level of the outer body — these are outer-locals
    # and capturing them is the closure case we must reject.
    outer_local_names: set = set(outer_param_names)
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    outer_local_names.add(tgt.id)
        elif isinstance(stmt, ast.FunctionDef):
            outer_local_names.add(stmt.name)

    new_body: List[ast.stmt] = []
    lifted: List[Tuple[str, str]] = []

    for stmt in fn.body:
        if not isinstance(stmt, ast.FunctionDef):
            new_body.append(stmt)
            continue

        free = _free_names(stmt)
        captured_outer = free & outer_local_names
        free_unbound = free - outer_local_names - module_names

        if captured_outer or free_unbound:
            # Either a real closure, or references undefined names. Leave
            # in place; the existing transformer will produce a diagnostic.
            new_body.append(stmt)
            continue

        # Pure but missing annotations? Chip transformer requires both
        # param and return annotations. Without them, we can't safely lift
        # — leave in body and let the diagnostic guide the user.
        all_args_annotated = all(a.annotation is not None for a in stmt.args.args)
        if not all_args_annotated or stmt.returns is None:
            new_body.append(stmt)
            continue

        # Pure: extract as a top-level chip source. Chip bodies use
        # `return X`, but circuit-style nested defs often write
        # `_zinnia_result = X`. Translate that to `return X` for the chip.
        chip_body = _rewrite_zinnia_result_to_return(stmt.body)
        clean = ast.FunctionDef(
            name=stmt.name,
            args=stmt.args,
            body=chip_body,
            decorator_list=[],
            returns=stmt.returns,
            type_comment=getattr(stmt, "type_comment", None),
        )
        ast.copy_location(clean, stmt)
        chip_src = ast.unparse(clean)
        lifted.append((stmt.name, chip_src))

    if not lifted:
        return source, []

    fn.body = new_body if new_body else [ast.Pass()]
    return ast.unparse(tree), lifted
