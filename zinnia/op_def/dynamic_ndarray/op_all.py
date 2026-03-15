from typing import Tuple

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NumberValue
from zinnia.compile.type_sys import DTDescriptor, IntegerType, BooleanType
from zinnia.op_def.dynamic_ndarray.abstract_aggregator import DynamicAbstractAggregator


class DynamicNDArray_AllOp(DynamicAbstractAggregator):
    def get_signature(self) -> str:
        return "DynamicNDArray.all"

    @classmethod
    def get_name(cls) -> str:
        return "all"

    def get_result_dtype(self, element_dt: DTDescriptor):
        return IntegerType

    def is_allowed_ndarray_dtype(self, element_dt: DTDescriptor) -> bool:
        return element_dt == IntegerType or element_dt == BooleanType

    def aggregator_func(
        self,
        builder: IRBuilderInterface,
        lhs: NumberValue,
        lhs_i: NumberValue,
        rhs: NumberValue,
        rhs_i: NumberValue,
        dt: DTDescriptor,
    ) -> Tuple[NumberValue, NumberValue | None]:
        return builder.ir_logical_and(builder.op_bool_cast(lhs), builder.op_bool_cast(rhs)), None
