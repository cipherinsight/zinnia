"""Rewrite ``for`` loops over iterator builtins into index loops.

The transformer calls `desugar_for_iterator` on every ``for`` statement (of
circuits and chips alike) before translating it:

- ``for t in reversed(xs)`` becomes ``for k in range(len(xs) - 1, -1, -1):
  t = xs[k]``.
- ``for a, b in zip(xs, ys, ...)`` becomes ``for k in range(min(len(xs),
  len(ys), ...)): a = xs[k]; b = ys[k]`` (a single target is bound to the
  tuple), so iteration stops at the shortest input, as ``zip`` does.

(``itertools.repeat`` used to be rewritten here too, matched by the name
``repeat`` alone, which also caught ``np.repeat(a, k)`` and iterated ``k``
times over the whole array. ``itertools`` is outside the import rule, so
that rewrite was removed; ``np.repeat`` is an ordinary array.)

The iterable expression is copied into the loop body, so it is evaluated
once per use rather than once before the loop; this is the same value for
the expressions these builtins are applied to in practice (names,
subscripts, ``range`` calls).
"""
from __future__ import annotations

import ast
from typing import Callable

from zinnia.compile.lowering.syntax import call, load, store, subscript


def _index_loop(original: ast.For, index: str, iterations: ast.expr, body: list[ast.stmt]) -> ast.For:
    loop = ast.For(target=store(index), iter=iterations, body=body, orelse=original.orelse)
    ast.copy_location(loop, original)
    ast.fix_missing_locations(loop)
    return loop


def desugar_for_iterator(node: ast.For, fresh_name: Callable[[str], str]) -> ast.For | None:
    """The index-loop form of ``node`` if it iterates over one of the
    builtins above, else ``None``. ``fresh_name(prefix)`` returns a new
    identifier for the loop index."""
    iterable = node.iter
    if not isinstance(iterable, ast.Call):
        return None
    if isinstance(iterable.func, ast.Name) and iterable.func.id == "reversed" and len(iterable.args) == 1:
        return _desugar_reversed(node, iterable.args[0], fresh_name("__rev_idx"))
    if isinstance(iterable.func, ast.Name) and iterable.func.id == "zip" and len(iterable.args) >= 2:
        return _desugar_zip(node, iterable.args, fresh_name("__zip_idx"))
    return None


def _desugar_reversed(node: ast.For, sequence: ast.expr, index: str) -> ast.For:
    minus_one = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))
    last = ast.BinOp(left=call("len", [sequence]), op=ast.Sub(), right=ast.Constant(value=1))
    iterations = call("range", [last, minus_one, ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))])
    element = ast.Assign(targets=[node.target], value=subscript(sequence, load(index)))
    return _index_loop(node, index, iterations, [element, *node.body])


def _desugar_zip(node: ast.For, sequences: list[ast.expr], index: str) -> ast.For:
    if isinstance(node.target, ast.Tuple) and len(node.target.elts) == len(sequences):
        bindings = [ast.Assign(targets=[target], value=subscript(sequence, load(index)))
                    for target, sequence in zip(node.target.elts, sequences)]
    else:
        items = ast.Tuple(elts=[subscript(sequence, load(index)) for sequence in sequences], ctx=ast.Load())
        bindings = [ast.Assign(targets=[node.target], value=items)]
    lengths = [call("len", [sequence]) for sequence in sequences]
    shortest = call("min", lengths)
    return _index_loop(node, index, call("range", [shortest]), [*bindings, *node.body])
