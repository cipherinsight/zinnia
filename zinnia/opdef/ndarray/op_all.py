from typing import Tuple

from zinnia.compile.type_sys import DTDescriptor, IntegerType
from zinnia.opdef.ndarray.abstract_aggregator import AbstractAggregator
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue


class NDArray_AllOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::all"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::all"

    def get_result_dtype(self, element_dt: DTDescriptor):
        return IntegerType

    def is_allowed_ndarray_dtype(self, element_dt: DTDescriptor) -> bool:
        return element_dt == IntegerType

    def aggregator_func(self, builder: AbsIRBuilderInterface, lhs: NumberValue, lhs_i: NumberValue, rhs: NumberValue, rhs_i: NumberValue, dt: DTDescriptor) -> Tuple[NumberValue, NumberValue | None]:
        return builder.ir_logical_and(lhs, rhs), None

    def initial_func(self, builder: AbsIRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        return builder.ir_constant_int(1), None
