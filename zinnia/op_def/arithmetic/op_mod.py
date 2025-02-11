from typing import Callable, Optional, Dict

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_arithemetic import AbstractArithemetic
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NumberValue, IntegerValue, FloatValue, NDArrayValue, ListValue, TupleValue, \
    Value


class ModOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "mod"

    @classmethod
    def get_name(cls) -> str:
        return "mod"

    def get_build_op_lambda(self, builder: IRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return builder.ir_mod_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return builder.ir_mod_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return builder.ir_mod_f(builder.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return builder.ir_mod_f(lhs, builder.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        lhs = kwargs['lhs']
        rhs = kwargs['rhs']
        if isinstance(lhs, NDArrayValue) and (isinstance(rhs, ListValue) or isinstance(rhs, TupleValue)):
            lhs, rhs = builder.op_implicit_type_align(lhs, rhs, dbg).values()
            return builder.op_modulo(lhs, rhs, dbg)
        elif (isinstance(lhs, ListValue) or isinstance(lhs, TupleValue)) and isinstance(rhs, NDArrayValue):
            lhs, rhs = builder.op_implicit_type_align(lhs, rhs, dbg).values()
            return builder.op_modulo(lhs, rhs, dbg)
        return super().build(builder, kwargs, dbg)
