from typing import Callable, Optional, Dict

from pyzk.debug.dbg_info import DebugInfo
from pyzk.opdef.nocls.abstract_compare import AbstractCompare
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import NumberValue, IntegerValue, FloatValue, Value, TupleValue, ListValue


class NotEqualOp(AbstractCompare):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "ne"

    @classmethod
    def get_name(cls) -> str:
        return "ne"

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_not_equal_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return reducer.ir_not_equal_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return reducer.ir_not_equal_f(reducer.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_not_equal_f(lhs, reducer.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, TupleValue) and isinstance(rhs, TupleValue):
            if lhs.types() != rhs.types():
                return reducer.ir_constant_int(1)
            result = reducer.ir_constant_int(0)
            for l, r in zip(lhs.values(), rhs.values()):
                result = reducer.ir_logical_or(result, reducer.op_bool_scalar(reducer.op_not_equal(l, r)))
            return result
        elif isinstance(lhs, ListValue) and isinstance(rhs, ListValue):
            if lhs.types() != rhs.types():
                return reducer.ir_constant_int(1)
            result = reducer.ir_constant_int(0)
            for l, r in zip(lhs.values(), rhs.values()):
                result = reducer.ir_logical_or(result, reducer.op_bool_scalar(reducer.op_not_equal(l, r)))
            return result
        return super().build(reducer, kwargs, dbg)
