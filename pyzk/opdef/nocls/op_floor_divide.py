from typing import Callable

from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import NumberValue, IntegerValue, FloatValue


class FloorDivideOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "floor_divide"

    @classmethod
    def get_name(cls) -> str:
        return "floor_divide"

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_floor_div_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_floor_div_f(lhs, reducer.ir_float_cast(rhs))
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return reducer.ir_floor_div_f(reducer.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return reducer.ir_floor_div_f(lhs, rhs)
            raise NotImplementedError()
        return _inner
