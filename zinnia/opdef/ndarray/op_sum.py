from typing import Tuple

from zinnia.compile.type_sys import DTDescriptor, IntegerDTDescriptor
from zinnia.opdef.ndarray.abstract_aggregator import AbstractAggregator
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue


class NDArray_SumOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::sum"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::sum"

    def aggregator_func(self, builder: AbsIRBuilderInterface, lhs: NumberValue, lhs_i: NumberValue, rhs: NumberValue, rhs_i: NumberValue, dt: DTDescriptor) -> Tuple[NumberValue, NumberValue | None]:
        return builder.op_add(lhs, rhs), None

    def initial_func(self, builder: AbsIRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        if isinstance(dt, IntegerDTDescriptor):
            return builder.ir_constant_int(0), None
        return builder.ir_constant_float(0.0), None
