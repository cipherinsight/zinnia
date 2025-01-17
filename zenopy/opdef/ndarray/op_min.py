from typing import Tuple

from zenopy.internal.dt_descriptor import DTDescriptor
from zenopy.opdef.ndarray.abstract_aggregator import AbstractAggregator
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import NumberValue


class NDArray_MinOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::min"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::min"

    def aggregator_func(self, reducer: AbsIRBuilderInterface, lhs: NumberValue, lhs_i: NumberValue, rhs: NumberValue, rhs_i: NumberValue, dt: DTDescriptor) -> Tuple[NumberValue, NumberValue | None]:
        cond = reducer.op_bool_scalar(reducer.op_less_than(lhs, rhs))
        return reducer.op_select(cond, lhs, rhs), None

    def initial_func(self, reducer: AbsIRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        return first_ele, None
