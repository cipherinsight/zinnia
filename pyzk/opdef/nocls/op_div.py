from typing import Callable

from pyzk.internal.dt_descriptor import DTDescriptor, FloatDTDescriptor
from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import NumberValue, IntegerValue


class DivOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "div"

    @classmethod
    def get_name(cls) -> str:
        return "div"

    def get_expected_result_dt(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor):
        return FloatDTDescriptor()

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue):
                lhs = reducer.ir_float_cast(lhs)
            if isinstance(rhs, IntegerValue):
                rhs = reducer.ir_float_cast(rhs)
            return reducer.ir_div_f(lhs, rhs)
        return _inner
