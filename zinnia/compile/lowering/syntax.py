"""Small AST helpers shared by the source-lowering passes."""
from __future__ import annotations

import ast
import copy
import textwrap


class FunctionLoweringError(ValueError):
    """A function, callback or import cannot be represented faithfully in a
    circuit, so the program is rejected instead of being compiled to
    something with different Python semantics."""


def dedent_source(source: str) -> str:
    """Remove the indentation every non-blank line shares.

    ``inspect.getsource`` of a method or of a function defined inside a test
    keeps the indentation of its enclosing block; ``ast.parse`` needs it gone.
    """
    lines = source.split("\n")
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    cut = min(indents) if indents else 0
    return "\n".join(line[cut:] for line in lines)


def load(identifier: str) -> ast.Name:
    return ast.Name(id=identifier, ctx=ast.Load())


def store(identifier: str) -> ast.Name:
    return ast.Name(id=identifier, ctx=ast.Store())


def call(function: str, args: list[ast.expr], keywords=()) -> ast.Call:
    """``function(*args, **keywords)`` for a function given by name."""
    return ast.Call(func=load(function), args=args, keywords=list(keywords))


def numpy_member(member: str) -> ast.Attribute:
    """``np.<member>``."""
    return ast.Attribute(value=load("np"), attr=member, ctx=ast.Load())


def subscript(value: ast.expr, index: ast.expr) -> ast.Subscript:
    return ast.Subscript(value=value, slice=index, ctx=ast.Load())


def is_numpy_call(node: ast.AST, member: str) -> bool:
    """Whether ``node`` is a call of ``np.<member>``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr == member
    )


def is_immutable_literal(value) -> bool:
    """Whether a value from ``ast.literal_eval`` can be shared safely: a
    number, string, bytes, ``None``, or a tuple of those."""
    if isinstance(value, (int, float, bool, str, bytes, type(None))):
        return True
    return isinstance(value, tuple) and all(is_immutable_literal(item) for item in value)


RESULT_NAME = "_zinnia_result"


def result_assignments_to_returns(body: list[ast.stmt]) -> list[ast.stmt]:
    """Turn top-level ``_zinnia_result = X`` statements into ``return X``.

    A circuit may give its result by assigning ``_zinnia_result`` instead of
    returning it; a local function written that way is a chip once lifted,
    and chips return their result.
    """
    rewritten = []
    for statement in body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == RESULT_NAME
        ):
            statement = ast.copy_location(ast.Return(value=statement.value), statement)
        rewritten.append(statement)
    return rewritten


def names_bound_by_target(target: ast.expr) -> list[str]:
    """The names an assignment or loop target binds (``a``, ``a, (b, c)``)."""
    return [node.id for node in ast.walk(target) if isinstance(node, ast.Name)]


def instantiate(template: str, substitutions: dict[str, str | ast.expr]) -> list[ast.stmt]:
    """Parse a statement template and fill in its placeholders.

    Placeholders are upper-case names in ``template``. A string substitution
    renames the placeholder (used for fresh, collision-free identifiers); an
    expression substitution replaces it with a copy of that expression. Only
    names are substituted, so the shape of the generated code is fixed by the
    template text.
    """
    tree = ast.parse(textwrap.dedent(template))

    class _Fill(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id not in substitutions:
                return node
            replacement = substitutions[node.id]
            if isinstance(replacement, str):
                return ast.Name(id=replacement, ctx=node.ctx)
            return ast.copy_location(copy.deepcopy(replacement), node)

    return [_Fill().visit(statement) for statement in tree.body]
