"""Which names a local function binds, and which it reads from outside.

Used by the helper lifter (`helpers.py`) to decide what a lifted local
function captures. The rules follow Python's scoping: a name is local to a
function if the function binds it anywhere in its body (assignment target,
``del``, nested ``def``), whatever the order; lambdas and comprehensions
have their own scopes, so their targets bind nothing outside them.
"""
from __future__ import annotations

import ast


def parameter_names(arguments: ast.arguments) -> set[str]:
    """Every parameter name, including ``*args`` and ``**kwargs``."""
    names = {a.arg for a in arguments.posonlyargs + arguments.args + arguments.kwonlyargs}
    if arguments.vararg:
        names.add(arguments.vararg.arg)
    if arguments.kwarg:
        names.add(arguments.kwarg.arg)
    return names


class _LocalBindings(ast.NodeVisitor):
    """Collects names a function body binds in the function's own scope."""

    def __init__(self, bound: set[str]):
        self.bound = bound

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # The definition binds its name; its body is a scope of its own.
        self.bound.add(node.name)

    def _own_scope(self, node: ast.AST) -> None:
        pass

    visit_Lambda = _own_scope
    visit_ListComp = _own_scope
    visit_GeneratorExp = _own_scope
    visit_SetComp = _own_scope
    visit_DictComp = _own_scope


def bound_names(function: ast.FunctionDef) -> set[str]:
    """The names local to ``function``: its parameters and every name its
    body binds."""
    bound = parameter_names(function.args)
    collector = _LocalBindings(bound)
    for statement in function.body:
        collector.visit(statement)
    return bound


class _FreeReads(ast.NodeVisitor):
    """Collects names read but not bound, tracking comprehension scopes."""

    def __init__(self, bound: set[str]):
        self.bound = bound
        self.free: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id not in self.bound:
            self.free.add(node.id)

    def _skip(self, node: ast.AST) -> None:
        # Nested functions and lambdas are lifted on their own; what they
        # read is captured when they are.
        pass

    visit_FunctionDef = _skip
    visit_Lambda = _skip

    def visit_Call(self, node: ast.Call) -> None:
        # A bare called name resolves through the compiler's chip, external
        # and builtin registry, so it is not a captured value.
        if not isinstance(node.func, ast.Name):
            self.visit(node.func)
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def _comprehension(self, node: ast.ListComp | ast.GeneratorExp | ast.SetComp | ast.DictComp) -> None:
        outer = set(self.bound)
        for generator in node.generators:
            self.visit(generator.iter)
            self.bound.update(n.id for n in ast.walk(generator.target) if isinstance(n, ast.Name))
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.bound.clear()
        self.bound.update(outer)

    visit_ListComp = _comprehension
    visit_GeneratorExp = _comprehension
    visit_SetComp = _comprehension
    visit_DictComp = _comprehension


def free_names(function: ast.FunctionDef) -> set[str]:
    """Names ``function`` reads from enclosing scopes (or builtins).

    Names read only inside nested functions or lambdas, and names only
    called (``f(x)`` reads ``x`` but not ``f``), are not included.
    """
    reads = _FreeReads(bound_names(function))
    for statement in function.body:
        reads.visit(statement)
    return reads.free
