from typing import Tuple

from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.opdef.ndarray.abstract_aggregator import AbstractAggregator
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue


class NDArray_MaxOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::max"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::max"

    def aggregator_func(self, builder: AbsIRBuilderInterface, lhs: NumberValue, lhs_i: NumberValue, rhs: NumberValue, rhs_i: NumberValue, dt: DTDescriptor) -> Tuple[NumberValue, NumberValue | None]:
        cond = builder.op_bool_scalar(builder.op_less_than(lhs, rhs))
        return builder.op_select(cond, rhs, lhs), None

    def initial_func(self, builder: AbsIRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        return first_ele, None
