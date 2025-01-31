from typing import Tuple

from zinnia.compile.type_sys import DTDescriptor, IntegerType
from zinnia.opdef.abstract.abstract_aggregator import AbstractAggregator
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue


class NDArray_ArgMaxOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.argmax"

    @classmethod
    def get_name(cls) -> str:
        return "argmax"

    def get_result_dtype(self, element_dt: DTDescriptor):
        return IntegerType

    def aggregator_func(self, builder: AbsIRBuilderInterface, lhs: NumberValue, lhs_i: NumberValue, rhs: NumberValue, rhs_i: NumberValue, dt: DTDescriptor) -> Tuple[NumberValue, NumberValue | None]:
        cond = builder.op_bool_cast(builder.op_less_than(lhs, rhs))
        candidate = builder.op_select(cond, rhs, lhs)
        candidate_i = builder.op_select(cond, rhs_i, lhs_i)
        return candidate, candidate_i

    def initial_func(self, builder: AbsIRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        return first_ele, builder.ir_constant_int(0)

    def depair_func(self, builder: AbsIRBuilderInterface, lhs: NumberValue, rhs: NumberValue) -> NumberValue:
        return rhs

    def enpair_func(self, builder: AbsIRBuilderInterface, a: NumberValue, b: int) -> Tuple[NumberValue, NumberValue | None]:
        return a, builder.ir_constant_int(b)
