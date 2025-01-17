from typing import Callable, Dict, Optional

from pyzk.debug.dbg_info import DebugInfo
from pyzk.debug.exception import StaticInferenceError
from pyzk.opdef.nocls.abstract_arithemetic import AbstractArithemetic
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, TupleValue, ListValue, NumberValue, IntegerValue, FloatValue


class MulOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "mul"

    @classmethod
    def get_name(cls) -> str:
        return "mul"

    def get_reduce_op_lambda(self, reducer: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_mul_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return reducer.ir_mul_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return reducer.ir_mul_f(reducer.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return reducer.ir_mul_f(lhs, reducer.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, IntegerValue) and isinstance(rhs, TupleValue):
            if lhs.val() is None:
                raise StaticInferenceError(dbg, f"Cannot statically inference the number of repetitions")
            return TupleValue(
                rhs.types() * lhs.val(),
                rhs.values() * lhs.val()
            )
        elif isinstance(lhs, TupleValue) and isinstance(rhs, IntegerValue):
            if rhs.val() is None:
                raise StaticInferenceError(dbg, f"Cannot statically inference the number of repetitions")
            return TupleValue(
                lhs.types() * rhs.val(),
                lhs.values() * rhs.val()
            )
        elif isinstance(lhs, IntegerValue) and isinstance(rhs, ListValue):
            if lhs.val() is None:
                raise StaticInferenceError(dbg, f"Cannot statically inference the number of repetitions")
            return ListValue(
                rhs.types() * lhs.val(),
                rhs.values() * lhs.val()
            )
        elif isinstance(lhs, ListValue) and isinstance(rhs, IntegerValue):
            if rhs.val() is None:
                raise StaticInferenceError(dbg, f"Cannot statically inference the number of repetitions")
            return ListValue(
                lhs.types() * rhs.val(),
                lhs.values() * rhs.val()
            )
        return super().build(reducer, kwargs, dbg)
