from typing import Tuple

from zinnia.compile.type_sys import DTDescriptor, IntegerType
from zinnia.op_def.abstract.abstract_aggregator import AbstractAggregator
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NumberValue


class NDArray_AnyOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.any"

    @classmethod
    def get_name(cls) -> str:
        return "any"

    def get_result_dtype(self, element_dt: DTDescriptor):
        return IntegerType

    def is_allowed_ndarray_dtype(self, element_dt: DTDescriptor) -> bool:
        return element_dt == IntegerType or element_dt == BooleanType

    def aggregator_func(self, builder: IRBuilderInterface, lhs: NumberValue, lhs_i: NumberValue, rhs: NumberValue, rhs_i: NumberValue, dt: DTDescriptor) -> Tuple[NumberValue, NumberValue | None]:
        return builder.ir_logical_or(builder.op_bool_cast(lhs), builder.op_bool_cast(rhs)), None

    def initial_func(self, builder: IRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        return builder.ir_constant_bool(False), None
