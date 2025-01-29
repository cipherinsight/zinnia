from typing import Callable, Dict, Optional

from zinnia.debug.dbg_info import DebugInfo
from zinnia.opdef.nocls.abstract_compare import AbstractCompare
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue, IntegerValue, FloatValue, Value, TupleValue, ListValue, \
    NDArrayValue


class LessThanOrEqualOp(AbstractCompare):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "lte"

    @classmethod
    def get_name(cls) -> str:
        return "lte"

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return builder.ir_less_than_or_equal_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return builder.ir_less_than_or_equal_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return builder.ir_less_than_or_equal_f(builder.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return builder.ir_less_than_or_equal_f(lhs, builder.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, TupleValue) and isinstance(rhs, TupleValue):
            result = builder.ir_constant_int(0)
            all_prev_eq = builder.ir_constant_int(1)
            for l, r in zip(lhs.values(), rhs.values()):
                lt_v = builder.op_bool_scalar(builder.op_less_than(l, r))
                eq_v = builder.op_bool_scalar(builder.op_equal(l, r))
                result = builder.ir_logical_or(result, builder.ir_logical_and(all_prev_eq, lt_v))
                all_prev_eq = builder.ir_logical_and(all_prev_eq, eq_v)
            if len(lhs.values()) < len(rhs.values()):
                return builder.ir_logical_or(result, all_prev_eq)
            return builder.ir_logical_or(result, builder.op_bool_scalar(builder.op_equal(lhs, rhs)))
        elif isinstance(lhs, ListValue) and isinstance(rhs, ListValue):
            result = builder.ir_constant_int(0)
            all_prev_eq = builder.ir_constant_int(1)
            for l, r in zip(lhs.values(), rhs.values()):
                lt_v = builder.op_bool_scalar(builder.op_less_than(l, r))
                eq_v = builder.op_bool_scalar(builder.op_equal(l, r))
                result = builder.ir_logical_or(result, builder.ir_logical_and(all_prev_eq, lt_v))
                all_prev_eq = builder.ir_logical_and(all_prev_eq, eq_v)
            if len(lhs.values()) < len(rhs.values()):
                return builder.ir_logical_or(result, all_prev_eq)
            return builder.ir_logical_or(result, builder.op_bool_scalar(builder.op_equal(lhs, rhs)))
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, ListValue):
            return builder.op_less_than_or_equal(lhs, builder.op_ndarray_asarray(rhs, dbg), dbg)
        elif isinstance(lhs, ListValue) and isinstance(rhs, NDArrayValue):
            return builder.op_less_than_or_equal(builder.op_ndarray_asarray(lhs, dbg), rhs, dbg)
        elif isinstance(lhs, NDArrayValue) and isinstance(rhs, TupleValue):
            return builder.op_less_than_or_equal(lhs, builder.op_ndarray_asarray(rhs, dbg), dbg)
        elif isinstance(lhs, TupleValue) and isinstance(rhs, NDArrayValue):
            return builder.op_less_than_or_equal(builder.op_ndarray_asarray(lhs, dbg), rhs, dbg)
        return super().build(builder, kwargs, dbg)
