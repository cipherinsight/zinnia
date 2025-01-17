from typing import Tuple

from zenopy.internal.dt_descriptor import DTDescriptor, IntegerType
from zenopy.opdef.ndarray.abstract_aggregator import AbstractAggregator
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import NumberValue


class NDArray_ArgMaxOp(AbstractAggregator):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::argmax"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::argmax"

    def get_result_dtype(self, element_dt: DTDescriptor):
        return IntegerType

    def aggregator_func(self, reducer: AbsIRBuilderInterface, lhs: NumberValue, lhs_i: NumberValue, rhs: NumberValue, rhs_i: NumberValue, dt: DTDescriptor) -> Tuple[NumberValue, NumberValue | None]:
        cond = reducer.op_bool_scalar(reducer.op_less_than(lhs, rhs))
        candidate = reducer.op_select(cond, rhs, lhs)
        candidate_i = reducer.op_select(cond, rhs_i, lhs_i)
        return candidate, candidate_i

    def initial_func(self, reducer: AbsIRBuilderInterface, dt: DTDescriptor, first_ele: NumberValue) -> Tuple[NumberValue, NumberValue | None]:
        return first_ele, reducer.ir_constant_int(0)

    def depair_func(self, reducer: AbsIRBuilderInterface, lhs: NumberValue, rhs: NumberValue) -> NumberValue:
        return rhs

    def enpair_func(self, reducer: AbsIRBuilderInterface, a: NumberValue, b: int) -> Tuple[NumberValue, NumberValue | None]:
        return a, reducer.ir_constant_int(b)
