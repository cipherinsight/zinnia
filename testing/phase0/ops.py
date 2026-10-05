"""The operator table of the reference evaluator (`acceptance.md` §2.1).

`src/phase0/ops.rs` maps every dispatched operator name to a *shape class*.
This module maps the same names to the NumPy function or Python builtin that
computes the actual value, so that `refeval.py` never reimplements NumPy shape
logic: it walks the AST and hands every operator to NumPy.

Three tables and nothing else:

| Table | Mirrors |
|---|---|
| `NUMPY` | `ops.rs::numpy`, the `np.*` / `zinnia.*` namespace |
| `MATH` | `ops.rs::math`, the `math.*` namespace |
| `BUILTINS` | `ops.rs::builtin`, a bare-name call |

A name absent from a table is an unsupported construct, which the harness
reports as a gap rather than as a phase 0 failure (`acceptance.md` §2.5).

Three Zinnia-specific semantics are implemented here rather than delegated,
because NumPy spells them differently (`coverage.md` §5.1, §5.2, §5.5):

* `np.where(m)` with one argument is the mask itself, as the baseline returns
  it and as phase 0's `E-Filter` reads it;
* `int`, `float` and `bool` are elementwise casts on an array, where Python
  would raise;
* `np.mean`, `np.var` and `np.std` fail on an empty axis, where NumPy returns
  `nan` with a warning: they divide by the count, and the baseline's every
  division carries `rhs != 0`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class UnsupportedConstruct(Exception):
    """The reference evaluator has no implementation for a construct.

    Raised only for harness gaps (`acceptance.md` §2.5); a genuine program
    error raises whatever NumPy or Python raises.
    """


@dataclass(frozen=True)
class DtypeClass:
    """A dtype object used as a value: `np.float64`, `Integer`, `int`.

    Phase 0 calls it `Desc::Class`; it has no shape. Calling it casts
    (`np.int64(x)`), which is `Class::Cast` in `ops.rs`.
    """

    name: str
    numpy_dtype: Any

    def __call__(self, value: Any) -> Any:
        return cast_to(value, self.numpy_dtype)


#: The names `Walker::register_global_datatypes` binds in every scope. `np` is
#: there too, as a bare `Value::Class`: it never carries a dtype, but it is one
#: of the 13 names the `if` merge of §6.1 visits, so it must be in the frame.
GLOBAL_DTYPES: dict[str, DtypeClass] = {
    name: DtypeClass(name, dtype)
    for name, dtype in [
        ("np", None),
        ("Float", np.float64),
        ("float", np.float64),
        ("Integer", np.int64),
        ("int", np.int64),
        ("Int", np.int64),
        ("integer", np.int64),
        # `bool` is `np.bool_`, not `np.int64`: the baseline aliases Boolean
        # to Integer (`coverage.md` §4) and phase 0 follows NumPy, so that
        # `np.ones(n, dtype=bool)` is a mask and `a[m]` is `E-Filter`
        # (`implementation.md`, "Consistency findings" 3).
        ("Boolean", np.bool_),
        ("bool", np.bool_),
        ("Bool", np.bool_),
        ("boolean", np.bool_),
        ("Complex", np.complex128),
        ("complex", np.complex128),
    ]
}

#: `np.<dtype>` names, collapsed by width exactly as `ops.rs::dtype_object` does.
DTYPE_OBJECTS: dict[str, Any] = {
    **{
        name: np.float64
        for name in ("float16", "float32", "float64", "double", "single", "half",
                     "longdouble")
    },
    **{
        name: np.int64
        for name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                     "uint64", "intp", "uintp", "long", "byte", "short", "intc",
                     "uintc", "ubyte", "ushort", "longlong", "ulonglong")
    },
    "bool_": np.bool_,
    **{
        name: np.complex128
        for name in ("complex64", "complex128", "complex256", "csingle", "cdouble",
                     "clongdouble")
    },
}


def cast_to(value: Any, dtype: Any) -> Any:
    """`x.astype(dt)`, `int(x)`, `np.float64(x)`: elementwise on an array,
    a scalar conversion otherwise (`coverage.md` §5.1).
    """
    if isinstance(value, np.ndarray):
        return value.astype(dtype)
    if isinstance(value, (list, tuple)):
        return type(value)(cast_to(item, dtype) for item in value)
    if dtype is np.bool_:
        return bool(value)
    if dtype is np.int64:
        return int(value)
    if dtype is np.float64:
        return float(value)
    return complex(value)


def _where(*args: Any) -> Any:
    """`np.where`. The one-argument form is the mask itself (`coverage.md`
    §5.5): the baseline returns `m`, so `a[np.where(m)]` is the mask read.
    """
    if len(args) == 1:
        return args[0]
    return np.where(*args)


def _averaging_reduction(name: str) -> Callable[..., Any]:
    """`np.mean`, `np.var` and `np.std`, which are undefined on an empty axis.

    NumPy returns `nan` with a warning. The three divide by the number of
    elements they reduce, and in the baseline every division carries
    `rhs != 0` as a requires (`src/optim/predicates/contracts.rs`), which a
    non-constant count turns into a runtime assertion — `np.mean` on a
    bounded-dynamic array asserts the length outright
    (`src/ops/dyn_ndarray/aggregation.rs`). So the run fails where NumPy
    warns, which is what `definition.md` §5.7 records as
    `defcond = (dims_a[k] > 0)`, and this is the value half of it.
    """
    reduce = getattr(np, name)

    def apply(a: Any, axis: Any = None, *rest: Any, **kwargs: Any) -> Any:
        if _reduced_count(np.asarray(a), axis) == 0:
            raise ZeroDivisionError(f"np.{name} over an empty axis")
        return reduce(a, axis, *rest, **kwargs)

    return apply


def _reduced_count(a: np.ndarray, axis: Any) -> int | None:
    """How many elements one cell of a reduction's output averages over.

    `None` where the axis is not one of `a`'s: NumPy has its own error for
    that and this wrapper must not pre-empt it.
    """
    if axis is None:
        return int(a.size)
    axes = axis if isinstance(axis, (tuple, list)) else (axis,)
    count = 1
    for entry in axes:
        k = int(entry)
        if not -a.ndim <= k < a.ndim:
            return None
        count *= a.shape[k]
    return count


def _outer(op: Callable[..., Any]) -> Callable[..., Any]:
    """`np.<binop>_outer`, the spelling the front end rewrites `np.add.outer`
    into (`coverage.md` §2).
    """

    def apply(a: Any, b: Any) -> Any:
        return op.outer(np.asarray(a), np.asarray(b))

    return apply


def _zeros_like_ctor(fill: Any) -> Callable[..., Any]:
    """`np.zeros(shape)` / `np.ones(shape)` / `np.empty(shape)`, whose shape
    argument may be an int, a list or a tuple.
    """

    def build(shape: Any, dtype: Any = np.float64, **_: Any) -> Any:
        return np.full(_as_shape(shape), fill, dtype=_as_dtype(dtype))

    return build


def _as_shape(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, (list, tuple, np.ndarray)):
        return tuple(int(n) for n in shape)
    return (int(shape),)


def _as_dtype(dtype: Any) -> Any:
    return dtype.numpy_dtype if isinstance(dtype, DtypeClass) else dtype


def _array(values: Any, dtype: Any = None, **_: Any) -> Any:
    return np.array(values, dtype=_as_dtype(dtype) if dtype is not None else None)


def _full(shape: Any, value: Any, dtype: Any = None, **_: Any) -> Any:
    return np.full(_as_shape(shape), value, dtype=_as_dtype(dtype) if dtype else None)


def _identity(n: Any, m: Any = None, k: int = 0, **_: Any) -> Any:
    """`np.identity(n)` and `np.eye(n, m, k)`. The baseline aliases `eye` to
    `identity` and drops `m` and `k` (`coverage.md` §9); NumPy reads them.
    """
    return np.eye(int(n), None if m is None else int(m), int(k))


#: `np.*` and `zinnia.*` (`ops.rs::numpy`).
NUMPY: dict[str, Callable[..., Any]] = {
    # elementwise, unary
    **{
        name: getattr(np, name)
        for name in ("sqrt", "exp", "log", "sin", "cos", "tan", "sinh", "cosh",
                     "tanh", "arccos", "arcsin", "arctan", "reciprocal", "negative",
                     "positive", "sign", "square", "floor", "ceil", "trunc", "round",
                     "deg2rad", "radians", "rad2deg", "degrees", "conj", "conjugate",
                     "absolute", "fabs", "real", "imag", "logical_not")
    },
    "acos": np.arccos,
    "asin": np.arcsin,
    "atan": np.arctan,
    "inv": lambda x: np.reciprocal(np.asarray(x, dtype=np.float64)),
    "abs": np.absolute,
    # elementwise, binary
    **{
        name: getattr(np, name)
        for name in ("add", "subtract", "multiply", "divide", "true_divide",
                     "floor_divide", "fmod", "remainder", "power", "equal",
                     "not_equal", "less", "less_equal", "greater", "greater_equal",
                     "logical_and", "logical_or", "logical_xor", "minimum",
                     "maximum", "fmin", "fmax", "arctan2", "clip", "isclose",
                     "allclose", "array_equal")
    },
    "mod": np.mod,
    "pow": np.power,
    "atan2": np.arctan2,
    "add_outer": _outer(np.add),
    "subtract_outer": _outer(np.subtract),
    "multiply_outer": _outer(np.multiply),
    "divide_outer": _outer(np.divide),
    # shape transforms
    **{
        name: getattr(np, name)
        for name in ("reshape", "ravel", "transpose", "swapaxes", "moveaxis",
                     "squeeze", "expand_dims", "broadcast_to", "atleast_1d",
                     "atleast_2d", "atleast_3d", "flip", "flipud", "fliplr", "rot90",
                     "concatenate", "stack", "vstack", "hstack", "dstack",
                     "column_stack", "diag", "delete", "insert", "pad", "repeat",
                     "tile", "diff", "sort", "argsort", "unique", "linspace",
                     "indices", "arange")
    },
    "flatten": np.ravel,
    "row_stack": np.vstack,
    # constructors
    "zeros": _zeros_like_ctor(0),
    "ones": _zeros_like_ctor(1),
    "empty": _zeros_like_ctor(0),
    "ndarray": _zeros_like_ctor(0),
    "full": _full,
    "zeros_like": np.zeros_like,
    "ones_like": np.ones_like,
    "empty_like": np.zeros_like,
    "full_like": np.full_like,
    "identity": _identity,
    "eye": _identity,
    "array": _array,
    "asarray": _array,
    "copy": np.copy,
    # reductions and statistics
    **{
        name: getattr(np, name)
        for name in ("sum", "prod", "min", "max", "amin", "amax", "any", "all",
                     "argmax", "argmin", "cumsum", "cumprod", "percentile")
    },
    **{name: _averaging_reduction(name) for name in ("mean", "var", "std")},
    # indexing and gather
    **{
        name: getattr(np, name)
        for name in ("take", "take_along_axis", "nonzero", "argwhere",
                     "flatnonzero", "bincount", "unravel_index",
                     "ravel_multi_index", "dot", "matmul", "tensordot",
                     "einsum", "corrcoef", "correlate")
    },
    "where": _where,
    "in1d": np.isin,
    "isin": np.isin,
    # Zinnia-specific
    "promote_to_dynamic": lambda a, **_: a,
    "pi": lambda: math.pi,
    "e": lambda: math.e,
}

#: `math.*` (`ops.rs::math`).
MATH: dict[str, Callable[..., Any]] = {
    **{
        name: getattr(math, name)
        for name in ("sin", "cos", "tan", "sinh", "cosh", "tanh", "sqrt", "exp",
                     "acos", "asin", "atan2", "floor", "ceil", "fabs")
    },
    "log": math.log,
    "inv": lambda x: 1.0 / x,
    "pi": lambda: math.pi,
    "e": lambda: math.e,
}

#: `np.linalg.*` (`ops.rs::linalg`).
LINALG: dict[str, Callable[..., Any]] = {"norm": np.linalg.norm}

#: The list methods of `S-ListMethod` (`ops.rs::list_method`).
LIST_METHODS = frozenset(
    {"append", "extend", "insert", "pop", "remove", "clear", "reverse", "copy",
     "count", "index"}
)

#: The list methods that change their receiver, and so rebind the name
#: (`walk/values.rs::list_method`). `copy` returns a new list, and `count` and
#: `index` read the elements, so those three leave the binding alone and take
#: no key of their own (`implementation.md` §5a rule 8).
REBINDING_LIST_METHODS = LIST_METHODS - {"copy", "count", "index"}

#: Builtins that iterate their first argument, so an *array* argument takes
#: one key per element before the call returns (§5a rule 4, and
#: `walk/values.rs`'s `builtin_enumerate` and `builtin_sequence`).
ITERATING_BUILTINS = frozenset({"enumerate", "list", "tuple"})

#: Names whose shape class is `NoShapeEffect` or `Copy` in `ops.rs`.
NO_SHAPE_EFFECT = frozenset({"print"})


def numpy_name(member: str) -> Callable[..., Any] | None:
    """The NumPy implementation of `np.<member>`, or `None`."""
    return NUMPY.get(member)


def builtin_call(member: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
    """A bare-name call (`ops.rs::builtin`)."""
    handler = BUILTINS.get(member)
    if handler is None:
        raise UnsupportedConstruct(f"builtin `{member}`")
    return handler(*args, **kwargs)


def _builtin_sum(iterable: Any, start: Any = 0) -> Any:
    """`sum(v)`: a flat sequence reduces to a scalar, a sequence of sequences
    along axis 0 (`values.rs::builtin_sum`).
    """
    array = np.asarray(iterable)
    return (array.sum(axis=0) if array.ndim >= 2 else array.sum()) + start


def _sequence(value: Any, kind: Callable[[Any], Any]) -> Any:
    """`list(v)` and `tuple(v)`: the same elements under the other kind.

    A value that is not a sequence comes back unchanged, which is what the
    baseline's `builtins.rs` does rather than raising
    (`values.rs::builtin_sequence`).
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return kind(value)
    return value


def _builtin_extremum(largest: bool) -> Callable[..., Any]:
    """`min(a, b, ...)` selects; `min(v)` reduces (`values.rs::builtin_minmax`)."""
    select = np.maximum if largest else np.minimum

    def apply(*args: Any) -> Any:
        if len(args) >= 2:
            result = args[0]
            for candidate in args[1:]:
                result = select(result, candidate)
            return result
        array = np.asarray(args[0])
        axis = 0 if array.ndim >= 2 else None
        return array.max(axis=axis) if largest else array.min(axis=axis)

    return apply


#: Bare-name calls (`ops.rs::builtin`).
BUILTINS: dict[str, Callable[..., Any]] = {
    "len": len,
    "range": lambda *a: list(range(*(int(x) for x in a))),
    "enumerate": lambda v: [(i, x) for i, x in enumerate(v)],
    "list": lambda v=(): _sequence(v, list),
    "tuple": lambda v=(): _sequence(v, tuple),
    "sum": _builtin_sum,
    "any": lambda v: bool(np.any(np.asarray(v))),
    "all": lambda v: bool(np.all(np.asarray(v))),
    "min": _builtin_extremum(largest=False),
    "max": _builtin_extremum(largest=True),
    "abs": np.absolute,
    "int": lambda x=0: cast_to(x, np.int64),
    "float": lambda x=0.0: cast_to(x, np.float64),
    "bool": lambda x=False: cast_to(x, np.bool_),
    "complex": lambda re=0.0, im=0.0: complex(re, im),
    "pow": lambda a, b: a**b,
    "print": lambda *a, **k: None,
}


# ---------------------------------------------------------------------------
# Binary and unary operators (`ops.rs::ZipOp`, `walk/exprs.rs`)
# ---------------------------------------------------------------------------


def is_sequence(value: Any) -> bool:
    """A Python list or tuple, i.e. phase 0's `Composite` (§3.3)."""
    return isinstance(value, (list, tuple))


def _is_numeric(value: Any) -> bool:
    """`value_ops.rs::is_numeric_composite`: every leaf is a number or array."""
    if isinstance(value, np.ndarray):
        return True
    if is_sequence(value):
        return all(_is_numeric(item) for item in value)
    return isinstance(value, (int, float, complex, bool, np.generic))


def _concat(lhs: Any, rhs: Any) -> Any:
    if isinstance(lhs, tuple) and isinstance(rhs, tuple):
        return lhs + rhs
    return list(lhs) + list(rhs)


def _repeat(lhs: Any, rhs: Any) -> Any:
    sequence, count = (lhs, rhs) if is_sequence(lhs) else (rhs, lhs)
    return sequence * int(count)


#: `a op b` once both sides are numbers or arrays (`ops.rs::ZipOp`).
NUMERIC_BINARY: dict[str, Callable[[Any, Any], Any]] = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "floor_div": np.floor_divide,
    "mod": np.mod,
    "pow": np.power,
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "lte": np.less_equal,
    "gt": np.greater,
    "gte": np.greater_equal,
    "bit_and": np.bitwise_and,
    "bit_or": np.bitwise_or,
    "bit_xor": np.bitwise_xor,
    "shl": np.left_shift,
    "shr": np.right_shift,
    "mat_mul": np.matmul,
}

#: Comparisons, which collapse to one Boolean on non-numeric sequences (§5.11).
COMPARISONS = frozenset({"eq", "ne", "lt", "lte", "gt", "gte"})
#: The connectives that read both sides as Booleans (`exprs.rs::scalar_connective`).
CONNECTIVES = frozenset({"and", "or"})


def binary(op: str, lhs: Any, rhs: Any) -> Any:
    """`lhs op rhs` with Zinnia's dispatch on the value kind (`coverage.md`
    §5.1): on composites `+` concatenates and `*` repeats, on anything with an
    array in it the operator is elementwise.
    """
    if op in ("in", "not_in"):
        member = any(bool(np.all(item == lhs)) for item in rhs)
        return member if op == "in" else not member
    if op in CONNECTIVES:
        return _connective(op, lhs, rhs)
    if op == "add" and is_sequence(lhs) and is_sequence(rhs):
        return _concat(lhs, rhs)
    if op == "mul" and is_sequence(lhs) != is_sequence(rhs):
        return _repeat(lhs, rhs)
    if op in COMPARISONS and is_sequence(lhs) and is_sequence(rhs):
        if not (_is_numeric(lhs) and _is_numeric(rhs)):
            return _PYTHON_COMPARISONS[op](lhs, rhs)
    apply = NUMERIC_BINARY.get(op)
    if apply is None:
        raise UnsupportedConstruct(f"binary operator `{op}`")
    return apply(_operand(lhs), _operand(rhs))


def _connective(op: str, lhs: Any, rhs: Any) -> Any:
    """`and` / `or`: elementwise on arrays, one Boolean on scalars.

    Zinnia reads both sides through `to_scalar_bool`, so the scalar result is
    a Boolean and not Python's "one of the operands".
    """
    if isinstance(lhs, np.ndarray) or isinstance(rhs, np.ndarray):
        combine = np.logical_and if op == "and" else np.logical_or
        return combine(lhs, rhs)
    return (bool(lhs) and bool(rhs)) if op == "and" else (bool(lhs) or bool(rhs))


def _operand(value: Any) -> Any:
    """A composite becomes the array it denotes, as `as_array` does."""
    return np.asarray(value) if is_sequence(value) else value


_PYTHON_COMPARISONS: dict[str, Callable[[Any, Any], Any]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
}


def unary(op: str, operand: Any) -> Any:
    """`+e`, `-e`, `~e`, `not e` (`exprs.rs::unary`)."""
    if op == "uadd":
        return operand
    if op == "usub":
        return np.negative(_operand(operand))
    if op == "invert":
        return np.invert(_operand(operand))
    if op == "not":
        if isinstance(operand, np.ndarray):
            return np.logical_not(operand)
        return not bool(operand)
    raise UnsupportedConstruct(f"unary operator `{op}`")


# ---------------------------------------------------------------------------
# Properties and methods (`E-Property` and `S-ListMethod`, §5.11)
# ---------------------------------------------------------------------------

#: `E-Property` (§5.11) plus the complex accessors next to it (`ops.rs::Prop`).
PROPERTIES: dict[str, Callable[[Any], Any]] = {
    "shape": lambda a: tuple(int(n) for n in np.asarray(a).shape),
    "ndim": lambda a: int(np.asarray(a).ndim),
    "size": lambda a: int(np.asarray(a).size),
    "dtype": lambda a: DtypeClass(str(np.asarray(a).dtype), np.asarray(a).dtype.type),
    "T": lambda a: np.transpose(np.asarray(a)),
    "flat": lambda a: np.ravel(np.asarray(a)),
}


#: Names whose shape argument may be spelled as several positional integers:
#: `a.reshape(2, 3)` and `np.transpose(a, 1, 0)` (`calls.rs::Args::rest`).
VARARG_SHAPE = frozenset({"reshape", "transpose", "broadcast_to", "tile"})


def collapse_varargs(member: str, args: list[Any]) -> list[Any]:
    """`f(a, 2, 3)` becomes `f(a, (2, 3))` for the names that accept both."""
    if member not in VARARG_SHAPE or len(args) < 3:
        return args
    tail = args[1:]
    if not all(isinstance(item, (int, np.integer)) and not isinstance(item, bool)
               for item in tail):
        return args
    return [args[0], tuple(int(item) for item in tail)]


def method_call(receiver: Any, member: str, args: list[Any],
                kwargs: dict[str, Any]) -> Any:
    """`x.m(...)`: the receiver becomes argument 0 of the `np.*` function of
    the same name, which is how the baseline's method table is built
    (`ops.rs::method`).
    """
    if member in PROPERTIES and not args:
        return PROPERTIES[member](receiver)
    if member == "astype":
        return cast_to(receiver, _as_dtype(args[0]) if args else np.int64)
    if member == "tolist":
        return np.asarray(receiver).tolist()
    if member == "filter":
        return np.asarray(receiver)[np.asarray(args[0]).astype(bool)]
    if member == "copy":
        return np.copy(receiver) if isinstance(receiver, np.ndarray) else list(receiver)
    if member in ("count", "index") and is_sequence(receiver):
        # `S-ListMethod`'s two reading methods (§5.11); they leave the
        # receiver alone, so they need no rebinding.
        return getattr(receiver, member)(args[0])
    handler = NUMPY.get(member)
    if handler is None:
        raise UnsupportedConstruct(f"method `.{member}()`")
    positional = collapse_varargs(member, [receiver, *args])
    return handler(*positional, **kwargs)
