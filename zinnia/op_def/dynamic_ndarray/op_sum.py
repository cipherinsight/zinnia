from typing import Tuple

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NumberValue
from zinnia.compile.type_sys import DTDescriptor, IntegerDTDescriptor
from zinnia.op_def.dynamic_ndarray.abstract_aggregator import DynamicAbstractAggregator


class DynamicNDArray_SumOp(DynamicAbstractAggregator):
    def get_signature(self) -> str:
        return "DynamicNDArray.sum"

    @classmethod
    def get_name(cls) -> str:
        return "sum"

    def aggregator_func(
        self,
        builder: IRBuilderInterface,
        lhs: NumberValue,
        lhs_i: NumberValue,
        rhs: NumberValue,
        rhs_i: NumberValue,
        dt: DTDescriptor,
    ) -> Tuple[NumberValue, NumberValue | None]:
        return builder.op_add(lhs, rhs), None
