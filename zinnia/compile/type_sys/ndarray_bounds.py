from dataclasses import dataclass
import time
from typing import Tuple

from z3 import z3

from zinnia.compile.triplet import IntegerValue, TupleValue
from zinnia.compile.type_sys import IntegerType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import StaticInferenceError, TypeInferenceError


def _accumulate_smt_time(elapsed_s: float) -> None:
    # Import lazily to avoid creating a hard import cycle at module import time.
    try:
        from zinnia.compile.builder.builder_impl import SMTUtils

        setattr(SMTUtils, "ACCUMULATED_TIME", getattr(SMTUtils, "ACCUMULATED_TIME", 0.0) + elapsed_s)
    except Exception:
        # Timing should never break compilation behavior.
        return


def _record_smt_invocation(assertion_count: int, timed_out: bool = False) -> None:
    # Import lazily to avoid creating a hard import cycle at module import time.
    try:
        from zinnia.compile.builder.builder_impl import SMTUtils

        SMTUtils.NUMBER_OF_CONSTRAINTS.append(assertion_count)
        if timed_out:
            setattr(SMTUtils, "NO_TIMEOUT_CASES", getattr(SMTUtils, "NO_TIMEOUT_CASES", 0) + 1)
    except Exception:
        # Metrics should never break compilation behavior.
        return


@dataclass(frozen=True)
class NDArrayCompileBounds:
    static_shape: Tuple[int, ...]
    max_rank: int
    max_length: int


def infer_ndarray_compile_bounds_from_shape(
    shape: TupleValue,
    builder,
    dbg: DebugInfo | None,
    opname: str,
) -> NDArrayCompileBounds:
    if not isinstance(shape, TupleValue):
        raise TypeInferenceError(dbg, f"`shape` of `{opname}` must be a Tuple")

    parsed_shape = []
    for ele_t, ele_v in zip(shape.types(), shape.values()):
        if ele_t != IntegerType:
            raise TypeInferenceError(dbg, f"Every element in `shape` of `{opname}` must be an Integer")
        assert isinstance(ele_v, IntegerValue)
        dim = ele_v.val(builder)
        if dim is None:
            raise StaticInferenceError(
                dbg,
                f"Cannot infer compile-time bound for `shape` in `{opname}`. "
                f"Maximum length/rank must be SMT-inferrable compile-time constants.",
            )
        if dim <= 0:
            raise TypeInferenceError(dbg, f"Every element in `shape` of `{opname}` must be greater than 0")
        parsed_shape.append(dim)

    max_rank = len(parsed_shape)
    max_length = 1
    for dim in parsed_shape:
        max_length *= dim

    return NDArrayCompileBounds(tuple(parsed_shape), max_rank, max_length)


def infer_ndarray_compile_bounds_from_static_shape(
    shape: Tuple[int, ...],
    dbg: DebugInfo | None,
    opname: str,
) -> NDArrayCompileBounds:
    if not isinstance(shape, tuple):
        raise TypeInferenceError(dbg, f"`shape` of `{opname}` must be a tuple")

    for dim in shape:
        if not isinstance(dim, int):
            raise TypeInferenceError(
                dbg,
                f"Cannot infer compile-time bound for `{opname}`. "
                f"Shape entries must be statically inferrable integers.",
            )
        if dim <= 0:
            raise TypeInferenceError(dbg, f"Every dimension in `{opname}` must be greater than 0")

    max_rank = len(shape)
    max_length = 1
    for dim in shape:
        max_length *= dim

    return NDArrayCompileBounds(shape, max_rank, max_length)


def infer_ndarray_max_bounds_from_shape(
    shape: TupleValue,
    builder,
    dbg: DebugInfo | None,
    opname: str,
) -> NDArrayCompileBounds:
    if not isinstance(shape, TupleValue):
        raise TypeInferenceError(dbg, f"`shape` of `{opname}` must be a Tuple")

    max_rank = len(shape.values())
    if max_rank <= 0:
        raise TypeInferenceError(dbg, f"`shape` of `{opname}` must be non-empty")

    dim_exprs = []
    constraints = []
    for ele_t, ele_v in zip(shape.types(), shape.values()):
        if ele_t != IntegerType:
            raise TypeInferenceError(dbg, f"Every element in `shape` of `{opname}` must be an Integer")
        assert isinstance(ele_v, IntegerValue)
        dim = ele_v.val(builder)
        if dim is not None:
            if dim <= 0:
                raise TypeInferenceError(dbg, f"Every element in `shape` of `{opname}` must be greater than 0")
            dim_exprs.append(z3.IntVal(dim))
            continue

        ptr = ele_v.ptr()
        if ptr is None:
            raise StaticInferenceError(
                dbg,
                f"Cannot infer compile-time maximum bound for `shape` in `{opname}`.",
            )
        dim_sym = z3.Int(f"int_{ptr}")
        dim_exprs.append(dim_sym)
        constraints.append(dim_sym > 0)
        if hasattr(builder, "_build_smt_constraints_for"):
            constraints.extend(builder._build_smt_constraints_for(ptr))

    product_expr = z3.IntVal(1)
    for dim_expr in dim_exprs:
        product_expr = product_expr * dim_expr

    opt = z3.Optimize()
    opt.set(timeout=1000)
    for c in constraints:
        opt.add(c)
    handle = opt.maximize(product_expr)
    solve_start = time.time()
    try:
        solve_res = opt.check()
    finally:
        _accumulate_smt_time(time.time() - solve_start)
    timed_out = solve_res == z3.unknown and "timeout" in str(opt.reason_unknown())
    _record_smt_invocation(len(opt.assertions()), timed_out)
    if solve_res != z3.sat:
        raise StaticInferenceError(
            dbg,
            f"Cannot infer compile-time maximum bound for `shape` in `{opname}`. "
            f"Maximum flattened length must be SMT-inferrable.",
        )

    upper = opt.upper(handle)
    upper_text = str(upper)
    if upper_text in {"oo", "+oo", "-oo"}:
        raise StaticInferenceError(
            dbg,
            f"Cannot infer compile-time maximum bound for `shape` in `{opname}` because bound is unbounded.",
        )
    try:
        max_length = int(upper_text)
    except ValueError:
        model = opt.model()
        max_length = int(str(model.eval(product_expr, model_completion=True)))

    if max_length <= 0:
        raise StaticInferenceError(
            dbg,
            f"Cannot infer positive maximum flattened length for `shape` in `{opname}`.",
        )

    # Dynamic arrays currently use bounded flat storage while rank is carried in runtime metadata.
    return NDArrayCompileBounds((max_length,), max_rank, max_length)
