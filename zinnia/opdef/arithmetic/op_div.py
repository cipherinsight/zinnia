from typing import Callable, Dict, Optional

from zinnia.compile.type_sys import DTDescriptor, FloatDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.opdef.abstract.abstract_arithemetic import AbstractArithemetic
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue, IntegerValue, NDArrayValue, ListValue, TupleValue, Value


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

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue):
                lhs = builder.ir_float_cast(lhs)
            if isinstance(rhs, IntegerValue):
                rhs = builder.ir_float_cast(rhs)
            return builder.ir_div_f(lhs, rhs)
        return _inner

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs = kwargs['lhs']
        rhs = kwargs['rhs']
        if isinstance(lhs, NDArrayValue) and (isinstance(rhs, ListValue) or isinstance(rhs, TupleValue)):
            lhs, rhs = builder.op_implicit_type_align(lhs, rhs, dbg).values()
            kwargs['lhs'] = lhs
            kwargs['rhs'] = rhs
            return super().build(builder, kwargs, dbg)
        elif (isinstance(lhs, ListValue) or isinstance(lhs, TupleValue)) and isinstance(rhs, NDArrayValue):
            lhs, rhs = builder.op_implicit_type_align(lhs, rhs, dbg).values()
            kwargs['lhs'] = lhs
            kwargs['rhs'] = rhs
            return super().build(builder, kwargs, dbg)
        return super().build(builder, kwargs, dbg)
