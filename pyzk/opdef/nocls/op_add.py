from typing import Callable, Dict, Optional

from pyzk.debug.dbg_info import DebugInfo
from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import NumberValue, IntegerValue, FloatValue, Value, TupleValue, ListValue


class AddOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "add"

    @classmethod
    def get_name(cls) -> str:
        return "add"

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_add_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return reducer.ir_add_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return reducer.ir_add_f(reducer.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_add_f(lhs, reducer.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, TupleValue) and isinstance(rhs, TupleValue):
            return TupleValue(
                lhs.types() + rhs.types(),
                lhs.values() + rhs.values()
            )
        elif isinstance(lhs, ListValue) and isinstance(rhs, ListValue):
            return ListValue(
                lhs.types() + rhs.types(),
                lhs.values() + rhs.values()
            )
        return super().build(reducer, kwargs, dbg)
