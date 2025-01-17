from typing import Callable

from zenopy.opdef.nocls.abstract_arithemetic import AbstractArithemetic
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import IntegerValue, NumberValue, FloatValue


class MinimumOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "minimum"

    @classmethod
    def get_name(cls) -> str:
        return "minimum"

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return reducer.op_min(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return reducer.op_min(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return reducer.op_min(reducer.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return reducer.op_min(lhs, reducer.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner
