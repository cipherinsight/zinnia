from typing import Tuple

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NumberValue
from zinnia.compile.type_sys import DTDescriptor, IntegerType
from zinnia.op_def.dynamic_ndarray.abstract_aggregator import DynamicAbstractAggregator


class DynamicNDArray_ArgMinOp(DynamicAbstractAggregator):
    def get_signature(self) -> str:
        return "DynamicNDArray.argmin"

    @classmethod
    def get_name(cls) -> str:
        return "argmin"

    def get_result_dtype(self, element_dt: DTDescriptor):
        return IntegerType

    def aggregator_func(
        self,
        builder: IRBuilderInterface,
        lhs: NumberValue,
        lhs_i: NumberValue,
        rhs: NumberValue,
        rhs_i: NumberValue,
        dt: DTDescriptor,
    ) -> Tuple[NumberValue, NumberValue | None]:
        cond = builder.op_bool_cast(builder.op_greater_than(lhs, rhs))
        candidate = builder.op_select(cond, rhs, lhs)
        candidate_i = builder.op_select(cond, rhs_i, lhs_i)
        return candidate, candidate_i

    def depair_func(self, builder: IRBuilderInterface, a: NumberValue, b: NumberValue) -> NumberValue:
        return b
