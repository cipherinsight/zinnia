from typing import Callable, Dict, Optional

from zenopy.debug.dbg_info import DebugInfo
from zenopy.opdef.nocls.abstract_compare import AbstractCompare
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import NumberValue, IntegerValue, FloatValue, TupleValue, Value, ListValue


class GreaterThanOp(AbstractCompare):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "gt"

    @classmethod
    def get_name(cls) -> str:
        return "gt"

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_greater_than_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return reducer.ir_greater_than_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return reducer.ir_greater_than_f(reducer.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_greater_than_f(lhs, reducer.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, TupleValue) and isinstance(rhs, TupleValue):
            result = reducer.ir_constant_int(0)
            all_prev_eq = reducer.ir_constant_int(1)
            for l, r in zip(lhs.values(), rhs.values()):
                gt_v = reducer.op_bool_scalar(reducer.op_greater_than(l, r))
                eq_v = reducer.op_bool_scalar(reducer.op_equal(l, r))
                all_prev_eq = reducer.ir_logical_and(all_prev_eq, eq_v)
                reducer.ir_logical_or(result, reducer.ir_logical_and(all_prev_eq, gt_v))
            if len(lhs.values()) > len(rhs.values()):
                return reducer.ir_logical_or(result, all_prev_eq)
            return result
        elif isinstance(lhs, ListValue) and isinstance(rhs, ListValue):
            result = reducer.ir_constant_int(0)
            all_prev_eq = reducer.ir_constant_int(1)
            for l, r in zip(lhs.values(), rhs.values()):
                gt_v = reducer.op_bool_scalar(reducer.op_greater_than(l, r))
                eq_v = reducer.op_bool_scalar(reducer.op_equal(l, r))
                all_prev_eq = reducer.ir_logical_and(all_prev_eq, eq_v)
                reducer.ir_logical_or(result, reducer.ir_logical_and(all_prev_eq, gt_v))
            if len(lhs.values()) > len(rhs.values()):
                return reducer.ir_logical_or(result, all_prev_eq)
            return result
        return super().build(reducer, kwargs, dbg)
