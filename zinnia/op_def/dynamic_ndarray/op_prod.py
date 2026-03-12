from typing import Tuple

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NumberValue
from zinnia.compile.type_sys import DTDescriptor
from zinnia.op_def.dynamic_ndarray.abstract_aggregator import DynamicAbstractAggregator


class DynamicNDArray_ProdOp(DynamicAbstractAggregator):
    def get_signature(self) -> str:
        return "DynamicNDArray.prod"

    @classmethod
    def get_name(cls) -> str:
        return "prod"

    def aggregator_func(
        self,
        builder: IRBuilderInterface,
        lhs: NumberValue,
        lhs_i: NumberValue,
        rhs: NumberValue,
        rhs_i: NumberValue,
        dt: DTDescriptor,
    ) -> Tuple[NumberValue, NumberValue | None]:
        return builder.op_multiply(lhs, rhs), None
