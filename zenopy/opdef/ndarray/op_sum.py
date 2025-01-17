from typing import Tuple

from zenopy.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor
from zenopy.opdef.ndarray.abstract_aggregator import AbstractAggregator
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import NumberValue


class NDArray_SumOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::sum"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::sum"

    def aggregator_func(self, reducer: AbsIRBuilderInterface, lhs: NumberValue, lhs_i: NumberValue, rhs: NumberValue, rhs_i: NumberValue, dt: DTDescriptor) -> Tuple[NumberValue, NumberValue | None]:
        return reducer.op_add(lhs, rhs), None

    def initial_func(self, reducer: AbsIRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        if isinstance(dt, IntegerDTDescriptor):
            return reducer.ir_constant_int(0), None
        return reducer.ir_constant_float(0.0), None
