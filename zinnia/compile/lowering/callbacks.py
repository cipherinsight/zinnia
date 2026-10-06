"""Lower NumPy and Python APIs that take a function argument.

Each API call becomes a call of a generated helper chip whose body is the
loop NumPy or Python would run, written with operations the compiler
supports. The callback itself is a lifted helper (`helpers.py`), a lambda, or
a function the compiler already knows (a chip, a builtin, ``np.<ufunc>``).

The rewrites, each preserving the Python/NumPy rule stated:

- ``map(f, xs, ...)``, ``filter(f, xs)``, ``filter(None, xs)``: a generator
  expression over ``xs`` (``zip`` of the iterables for several), so the
  result stays a one-shot iterator; nested generators are fused.
- ``np.vectorize(f, otypes=...)(a)``: ``f`` applied to every element of
  ``a`` in row-major order, reshaped to ``a``'s shape. Without ``otypes``
  NumPy calls ``f`` on the first element to find the output dtype (and calls
  it again on that element unless ``cache=True``); an empty input is an
  error there, and is rejected here.
- ``np.piecewise(x, condlist, funclist)``: zeros, then for each condition in
  order ``out[cond] = f(x[cond])`` (later conditions win); an extra function
  covers the elements no condition selects. A function entry that is not
  callable is a scalar value.
- ``np.apply_along_axis(f, axis, a, *args, **kwargs)``: ``f`` applied to
  every 1-D slice along ``axis`` (a literal), results stacked in place of
  that axis; an input with no slices is rejected.
- ``np.fromiter(iterable, dtype, count=-1)``: the first ``count`` items
  (all items for a negative count); fewer items than ``count`` is an error.
"""
from __future__ import annotations

import ast
import builtins
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zinnia.compile.lowering.syntax import (
    FunctionLoweringError, call, instantiate, is_numpy_call, load, numpy_member, store, subscript,
)

if TYPE_CHECKING:
    from zinnia.compile.lowering.helpers import HelperLowering


@dataclass
class VectorizedCallback:
    """The value of ``np.vectorize(f, ...)``: called later with an array."""

    callback: object
    otypes: ast.expr | None = None
    cache: bool = False


def _comprehension(target: ast.expr, iterable: ast.expr, conditions: list[ast.expr]) -> ast.comprehension:
    return ast.comprehension(target=target, iter=iterable, ifs=conditions, is_async=0)


# --- np.vectorize ------------------------------------------------------------

def _is_single_output_type(otypes: ast.expr) -> bool:
    """``otypes`` names exactly one output dtype: a one-character type code
    or a one-element list or tuple."""
    if isinstance(otypes, ast.Constant):
        return isinstance(otypes.value, str) and len(otypes.value) == 1
    return isinstance(otypes, (ast.List, ast.Tuple)) and len(otypes.elts) == 1


def vectorize(lowering: HelperLowering, node: ast.Call, env: dict) -> VectorizedCallback:
    """Read ``np.vectorize(f, otypes=None, cache=False)``."""
    if not node.args:
        raise FunctionLoweringError('np.vectorize requires a callback')
    options = {keyword.arg: keyword.value for keyword in node.keywords}
    if len(node.args) > 2 or set(options) - {'otypes', 'cache'}:
        raise FunctionLoweringError('np.vectorize signature/excluded options are unsupported')
    cache = options.get('cache', ast.Constant(False))
    if not isinstance(cache, ast.Constant) or not isinstance(cache.value, bool):
        raise FunctionLoweringError('np.vectorize cache must be a literal bool')
    otypes = node.args[1] if len(node.args) > 1 else options.get('otypes')
    if isinstance(otypes, ast.Constant) and otypes.value is None:
        otypes = None
    if otypes is not None and not _is_single_output_type(otypes):
        raise FunctionLoweringError('np.vectorize lowering supports exactly one scalar output dtype')
    return VectorizedCallback(lowering.callback(node.args[0], env), otypes, cache.value)


def call_vectorized(lowering: HelperLowering, vectorized: VectorizedCallback,
                    args: list[ast.expr], keywords: list[ast.keyword], env: dict) -> ast.Call:
    """A call of a ``np.vectorize`` result with one array argument."""
    if len(args) != 1 or keywords:
        raise FunctionLoweringError('np.vectorize lowering currently accepts one array argument')
    array, flat, item, result = [lowering.fresh(stem) for stem in ('array', 'flat', 'item', 'result')]
    names = {'ARRAY': array, 'FLAT': flat}
    body = instantiate('FLAT = np.asarray(ARRAY).reshape(-1)', names)
    infer_dtype = vectorized.otypes is None
    if infer_dtype:
        body += instantiate('assert len(FLAT) > 0', names)
    first = lowering.fresh('first')
    if infer_dtype:
        first_value = lowering.invoke(vectorized.callback, [subscript(load(flat), ast.Constant(0))])
        body.append(ast.Assign(targets=[store(first)], value=first_value))
        # dtype=np.asarray(first).dtype
        dtype = ast.Attribute(value=ast.Call(func=numpy_member('asarray'), args=[load(first)], keywords=[]),
                              attr='dtype', ctx=ast.Load())
    else:
        dtype = subscript(copy.deepcopy(vectorized.otypes), ast.Constant(0))
    reuse_first = infer_dtype and vectorized.cache
    elements: ast.expr = load(flat)
    if reuse_first:
        elements = subscript(load(flat), ast.Slice(lower=ast.Constant(1)))
    values: ast.expr = ast.ListComp(
        elt=lowering.invoke(vectorized.callback, [load(item)]),
        generators=[_comprehension(store(item), elements, [])],
    )
    if reuse_first:
        values = ast.BinOp(left=ast.List(elts=[load(first)], ctx=ast.Load()), op=ast.Add(), right=values)
    array_value = ast.Call(func=numpy_member('array'), args=[values],
                           keywords=[ast.keyword(arg='dtype', value=dtype)])
    body.append(ast.Assign(targets=[store(result)], value=array_value))
    body += instantiate('return RESULT.reshape(np.asarray(ARRAY).shape)', {'RESULT': result, 'ARRAY': array})
    helper = lowering.generated_helper('vectorize', [array], body, env)
    return lowering.invoke(helper, args)


# --- np.piecewise ------------------------------------------------------------

def piecewise(lowering: HelperLowering, node: ast.Call, env: dict) -> ast.Call:
    if len(node.args) != 3 or node.keywords:
        raise FunctionLoweringError('np.piecewise lowering requires x, a literal condition list, and a literal function list')
    x, conditions, functions = node.args
    if not isinstance(conditions, (ast.List, ast.Tuple)) or not isinstance(functions, (ast.List, ast.Tuple)):
        raise FunctionLoweringError('np.piecewise conditions/functions must be syntactic lists or tuples')
    if len(functions.elts) not in (len(conditions.elts), len(conditions.elts) + 1):
        raise FunctionLoweringError('np.piecewise function count must equal condition count or exceed it by one')
    array, result, covered = [lowering.fresh(stem) for stem in ('array', 'result', 'covered')]
    masks = [lowering.fresh('mask') for _ in conditions.elts]
    parameters = [array] + masks
    arguments = [lowering.lower_expression(x, env)]
    arguments += [lowering.lower_expression(condition, env) for condition in conditions.elts]
    names = {'ARRAY': array, 'RESULT': result, 'COVERED': covered}
    body = instantiate('''
        RESULT = np.zeros_like(ARRAY)
        COVERED = np.zeros_like(ARRAY, dtype=bool)
    ''', names)
    for mask in masks:
        body += instantiate('COVERED = COVERED | MASK', {**names, 'MASK': mask})
    if len(functions.elts) > len(masks):
        otherwise = lowering.fresh('mask')
        body += instantiate('OTHERWISE = ~COVERED', {**names, 'OTHERWISE': otherwise})
        masks.append(otherwise)
    for mask, function in zip(masks, functions.elts):
        if _is_callable_entry(function, env):
            callback = lowering.callback(function, env)
            value = lowering.invoke(callback, [subscript(load(array), load(mask))])
            update = ast.Assign(targets=[ast.Subscript(value=load(result), slice=load(mask), ctx=ast.Store())],
                                value=value)
            # Only call the function when it selects something, as NumPy does.
            body.append(ast.If(test=ast.Call(func=numpy_member('any'), args=[load(mask)], keywords=[]),
                               body=[update], orelse=[]))
        else:
            scalar = lowering.fresh('choice')
            parameters.append(scalar)
            arguments.append(lowering.lower_expression(function, env))
            body.append(ast.Assign(targets=[ast.Subscript(value=load(result), slice=load(mask), ctx=ast.Store())],
                                   value=load(scalar)))
    body.append(ast.Return(value=load(result)))
    return lowering.invoke(lowering.generated_helper('piecewise', parameters, body, env), arguments)


def _is_callable_entry(function: ast.expr, env: dict) -> bool:
    """Whether a ``funclist`` entry is a function (else it is a value)."""
    if isinstance(function, (ast.Lambda, ast.Attribute)):
        return True
    return isinstance(function, ast.Name) and (
        function.id in env or callable(getattr(builtins, function.id, None)))


# --- np.apply_along_axis -----------------------------------------------------

_APPLY_ALONG_AXIS_PROLOGUE = '''
    AXIS = AXIS_VALUE
    if AXIS < 0:
        AXIS = AXIS + ARRAY.ndim
    assert 0 <= AXIS < ARRAY.ndim
    MOVED = np.moveaxis(ARRAY, AXIS, -1)
    ROWS = MOVED.reshape((-1, MOVED.shape[-1]))
    assert len(ROWS) > 0
'''

_APPLY_ALONG_AXIS_EPILOGUE = '''
    RESULT = np.array(VALUES)
    RESULT = RESULT.reshape(MOVED.shape[:-1] + RESULT.shape[1:])
    return RESULT.transpose(tuple(range(AXIS)) + tuple(range(MOVED.ndim - 1, RESULT.ndim)) + tuple(range(AXIS, MOVED.ndim - 1)))
'''


def apply_along_axis(lowering: HelperLowering, node: ast.Call, env: dict) -> ast.Call:
    if len(node.args) < 3:
        raise FunctionLoweringError('np.apply_along_axis requires callback, axis, and array positionally')
    callback = lowering.callback(node.args[0], env)
    try:
        axis = ast.literal_eval(node.args[1])
    except (ValueError, TypeError):
        raise FunctionLoweringError('np.apply_along_axis axis must be a literal integer')
    if not isinstance(axis, int):
        raise FunctionLoweringError('np.apply_along_axis axis must be a literal integer')
    array, normalized, moved, rows, index, values, result = [
        lowering.fresh(stem) for stem in ('array', 'axis', 'moved', 'rows', 'i', 'values', 'result')]
    extra_parameters = [lowering.fresh('arg') for _ in node.args[3:]]
    keyword_parameters = [lowering.fresh('kwarg') for _ in node.keywords]
    if any(keyword.arg is None for keyword in node.keywords):
        raise FunctionLoweringError('Expanded callback keyword arguments are unsupported')
    names = {'ARRAY': array, 'AXIS': normalized, 'MOVED': moved, 'ROWS': rows,
             'VALUES': values, 'RESULT': result, 'AXIS_VALUE': ast.Constant(axis)}
    body = instantiate(_APPLY_ALONG_AXIS_PROLOGUE, names)
    row = subscript(load(rows), load(index))
    value = lowering.invoke(
        callback,
        [row] + [load(name) for name in extra_parameters],
        [ast.keyword(arg=keyword.arg, value=load(name)) for keyword, name in zip(node.keywords, keyword_parameters)],
    )
    every_row = call('range', [call('len', [load(rows)])])
    body.append(ast.Assign(targets=[store(values)],
                           value=ast.ListComp(elt=value, generators=[_comprehension(store(index), every_row, [])])))
    body += instantiate(_APPLY_ALONG_AXIS_EPILOGUE, names)
    helper = lowering.generated_helper('apply_axis', [array] + extra_parameters + keyword_parameters, body, env)
    arguments = [lowering.lower_expression(argument, env) for argument in node.args[2:]]
    arguments += [lowering.lower_expression(keyword.value, env) for keyword in node.keywords]
    return lowering.invoke(helper, arguments)


# --- np.fromiter -------------------------------------------------------------

_FROMITER_BODY = '''
    RESULT = []
    if COUNT != 0:
        for ITEM in ITERABLE:
            RESULT.append(ITEM)
            if COUNT >= 0 and len(RESULT) == COUNT:
                break
    assert COUNT < 0 or len(RESULT) == COUNT
    return np.array(RESULT, dtype=DTYPE)
'''


def fromiter(lowering: HelperLowering, node: ast.Call, env: dict) -> ast.Call:
    if not node.args:
        raise FunctionLoweringError('np.fromiter requires an iterable')
    options = {keyword.arg: keyword.value for keyword in node.keywords}
    if len(node.args) > 3 or set(options) - {'dtype', 'count'}:
        raise FunctionLoweringError('Unsupported np.fromiter arguments')
    dtype = node.args[1] if len(node.args) > 1 else options.get('dtype')
    if dtype is None:
        raise FunctionLoweringError('np.fromiter requires dtype')
    count = node.args[2] if len(node.args) > 2 else options.get('count', ast.Constant(-1))
    iterable, count_name, dtype_name, result, item = [
        lowering.fresh(stem) for stem in ('iter', 'count', 'dtype', 'result', 'i')]
    # Circuit iterables are finite; taking items in a loop (instead of
    # indexing the iterable) also works for generators and keeps the count check.
    body = instantiate(_FROMITER_BODY, {'RESULT': result, 'COUNT': count_name, 'ITEM': item,
                                        'ITERABLE': iterable, 'DTYPE': dtype_name})
    helper = lowering.generated_helper('fromiter', [iterable, dtype_name, count_name], body, env)
    arguments = [lowering.lower_expression(node.args[0], env),
                 lowering.lower_expression(dtype, env),
                 lowering.lower_expression(count, env)]
    return lowering.invoke(helper, arguments)


# --- map / filter ------------------------------------------------------------

def map_or_filter(lowering: HelperLowering, node: ast.Call, env: dict) -> ast.GeneratorExp:
    kind = node.func.id
    if node.keywords or len(node.args) < 2 or kind == 'filter' and len(node.args) != 2:
        raise FunctionLoweringError('Unsupported map/filter call signature')
    function = node.args[0]
    callback = None
    if not (isinstance(function, ast.Constant) and function.value is None):
        callback = lowering.callback(function, env)
    if callback is None and kind == 'map':
        raise FunctionLoweringError('map requires a callable')
    items = [lowering.fresh('item') for _ in node.args[1:]]
    if len(items) == 1:
        iterable = lowering.lower_expression(node.args[1], env)
        target: ast.expr = store(items[0])
    else:
        iterable = call('zip', [lowering.lower_expression(argument, env) for argument in node.args[1:]])
        target = ast.Tuple(elts=[store(name) for name in items], ctx=ast.Store())
    if callback is not None:
        value = lowering.invoke(callback, [load(name) for name in items])
    else:  # filter(None, xs) keeps the truthy items
        value = load(items[0])
    element = value if kind == 'map' else load(items[0])
    conditions = [value] if kind == 'filter' else []
    if isinstance(iterable, ast.GeneratorExp) and len(items) == 1:
        return _fuse_with_generator(iterable, items[0], target, element, conditions)
    return ast.GeneratorExp(elt=element, generators=[_comprehension(target, iterable, conditions)])


def _fuse_with_generator(inner: ast.GeneratorExp, item: str, target: ast.expr,
                         element: ast.expr, conditions: list[ast.expr]) -> ast.GeneratorExp:
    """``map``/``filter`` over a generator expression, as one generator."""
    generators = copy.deepcopy(inner.generators)
    if isinstance(inner.elt, ast.Name):
        # The inner element is a plain name: substitute it for the item.
        class _SubstituteItem(ast.NodeTransformer):
            def visit_Name(self, name: ast.Name) -> ast.AST:
                if name.id == item and isinstance(name.ctx, ast.Load):
                    return copy.deepcopy(inner.elt)
                return name

        element = _SubstituteItem().visit(element)
        conditions = [_SubstituteItem().visit(condition) for condition in conditions]
        generators[-1].ifs.extend(conditions)
    else:
        # Bind the inner element to the item with a one-element loop.
        singleton = ast.Tuple(elts=[inner.elt], ctx=ast.Load())
        generators.append(_comprehension(target, singleton, conditions))
    return ast.GeneratorExp(elt=element, generators=generators)


def lower_callback_api(lowering: HelperLowering, node: ast.Call, env: dict) -> ast.expr | None:
    """Lower ``node`` if it calls one of the APIs above; ``None`` otherwise."""
    if is_numpy_call(node.func, 'vectorize'):
        vectorized = vectorize(lowering, node.func, env)
        arguments = [lowering.lower_expression(argument, env) for argument in node.args]
        keywords = [ast.keyword(arg=k.arg, value=lowering.lower_expression(k.value, env)) for k in node.keywords]
        return call_vectorized(lowering, vectorized, arguments, keywords, env)
    if is_numpy_call(node, 'piecewise'):
        return piecewise(lowering, node, env)
    if is_numpy_call(node, 'apply_along_axis'):
        return apply_along_axis(lowering, node, env)
    if is_numpy_call(node, 'fromiter'):
        return fromiter(lowering, node, env)
    if isinstance(node.func, ast.Name) and node.func.id in ('map', 'filter') and node.func.id not in env:
        return map_or_filter(lowering, node, env)
    return None
