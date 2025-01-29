from typing import Callable, Dict, Optional

from zinnia.debug.dbg_info import DebugInfo
from zinnia.opdef.nocls.abstract_arithemetic import AbstractArithemetic
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue, IntegerValue, FloatValue, Value, TupleValue, ListValue, \
    StringValue


class AddOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "add"

    @classmethod
    def get_name(cls) -> str:
        return "add"

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return builder.ir_add_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return builder.ir_add_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return builder.ir_add_f(builder.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return builder.ir_add_f(lhs, builder.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, TupleValue) and isinstance(rhs, TupleValue):
            return TupleValue(
                lhs.types() + rhs.types(),
                lhs.values() + rhs.values()
            )
        elif isinstance(lhs, ListValue) and isinstance(rhs, ListValue):
            return ListValue(
                lhs.types() + rhs.types(),
                lhs.values() + rhs.values()
            )
        elif isinstance(lhs, StringValue) and isinstance(rhs, StringValue):
            return builder.ir_add_str(lhs, rhs)
        return super().build(builder, kwargs, dbg)
