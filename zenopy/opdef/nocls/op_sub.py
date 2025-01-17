from typing import Callable

from zenopy.opdef.nocls.abstract_arithemetic import AbstractArithemetic
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import NumberValue, IntegerValue, FloatValue


class SubOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "sub"

    @classmethod
    def get_name(cls) -> str:
        return "sub"

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_sub_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return reducer.ir_sub_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return reducer.ir_sub_f(reducer.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_sub_f(lhs, reducer.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner
