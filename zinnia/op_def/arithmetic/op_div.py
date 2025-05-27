from typing import Callable, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.type_sys import DTDescriptor, FloatDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_arithemetic import AbstractArithemetic
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NumberValue, IntegerValue, NDArrayValue, ListValue, TupleValue, Value


class DivOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "div"

    @classmethod
    def get_name(cls) -> str:
        return "div"

    def get_expected_result_dt(self, lhs_dt: DTDescriptor, rhs_dt: DTDescriptor):
        return FloatDTDescriptor()

    def get_build_op_lambda(self, builder: IRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue):
                lhs = builder.ir_float_cast(lhs)
            if isinstance(rhs, IntegerValue):
                rhs = builder.ir_float_cast(rhs)
            return builder.ir_div_f(lhs, rhs)
        return _inner

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        lhs = kwargs['lhs']
        rhs = kwargs['rhs']
        if isinstance(lhs, NDArrayValue) and (isinstance(rhs, ListValue) or isinstance(rhs, TupleValue)):
            lhs, rhs = builder.op_implicit_type_align(lhs, rhs, dbg).values()
            return builder.op_divide(lhs, rhs, dbg)
        elif (isinstance(lhs, ListValue) or isinstance(lhs, TupleValue)) and isinstance(rhs, NDArrayValue):
            lhs, rhs = builder.op_implicit_type_align(lhs, rhs, dbg).values()
            return builder.op_divide(lhs, rhs, dbg)
        return super().build(builder, kwargs, dbg)
