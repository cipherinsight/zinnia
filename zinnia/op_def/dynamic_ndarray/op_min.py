from typing import Tuple

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NumberValue
from zinnia.compile.type_sys import DTDescriptor
from zinnia.op_def.dynamic_ndarray.abstract_aggregator import DynamicAbstractAggregator


class DynamicNDArray_MinOp(DynamicAbstractAggregator):
    def get_signature(self) -> str:
        return "DynamicNDArray.min"

    @classmethod
    def get_name(cls) -> str:
        return "min"

    def aggregator_func(
        self,
        builder: IRBuilderInterface,
        lhs: NumberValue,
        lhs_i: NumberValue,
        rhs: NumberValue,
        rhs_i: NumberValue,
        dt: DTDescriptor,
    ) -> Tuple[NumberValue, NumberValue | None]:
        cond = builder.op_bool_cast(builder.op_less_than(lhs, rhs))
        return builder.op_select(cond, lhs, rhs), None
