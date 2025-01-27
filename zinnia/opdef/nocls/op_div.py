from typing import Callable

from zinnia.compile.type_sys import DTDescriptor, FloatDTDescriptor
from zinnia.opdef.nocls.abstract_arithemetic import AbstractArithemetic
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue, IntegerValue


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

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue):
                lhs = builder.ir_float_cast(lhs)
            if isinstance(rhs, IntegerValue):
                rhs = builder.ir_float_cast(rhs)
            return builder.ir_div_f(lhs, rhs)
        return _inner
