from typing import Tuple

from zinnia.compile.type_sys import DTDescriptor, IntegerDTDescriptor
from zinnia.op_def.abstract.abstract_aggregator import AbstractAggregator
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NumberValue


class NDArray_SumOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.sum"

    @classmethod
    def get_name(cls) -> str:
        return "sum"

    def aggregator_func(self, builder: IRBuilderInterface, lhs: NumberValue, lhs_i: NumberValue, rhs: NumberValue, rhs_i: NumberValue, dt: DTDescriptor) -> Tuple[NumberValue, NumberValue | None]:
        return builder.op_add(lhs, rhs), None

    def initial_func(self, builder: IRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        if isinstance(dt, IntegerDTDescriptor):
            return builder.ir_constant_int(0), None
        return builder.ir_constant_float(0.0), None
