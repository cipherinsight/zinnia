from typing import Callable, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import StaticInferenceError
from zinnia.op_def.abstract.abstract_arithemetic import AbstractArithemetic
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, TupleValue, ListValue, NumberValue, IntegerValue, FloatValue, \
    StringValue, NDArrayValue


class MulOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "mul"

    @classmethod
    def get_name(cls) -> str:
        return "mul"

    def get_build_op_lambda(self, builder: IRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return builder.ir_mul_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return builder.ir_mul_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return builder.ir_mul_f(builder.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return builder.ir_mul_f(lhs, builder.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
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
        elif isinstance(lhs, StringValue) and isinstance(rhs, IntegerValue):
            if rhs.val() is None:
                raise StaticInferenceError(dbg, f"Cannot statically inference the number of repetitions")
            result = lhs
            for _ in range(rhs.val()):
                result = builder.ir_add_str(result, lhs)
            return result
        elif isinstance(lhs, IntegerValue) and isinstance(rhs, StringValue):
            if rhs.val() is None:
                raise StaticInferenceError(dbg, f"Cannot statically inference the number of repetitions")
            result = rhs
            for _ in range(lhs.val()):
                result = builder.ir_add_str(result, rhs)
            return result
        elif isinstance(lhs, NDArrayValue) and (isinstance(rhs, ListValue) or isinstance(rhs, TupleValue)):
            lhs, rhs = builder.op_implicit_type_align(lhs, rhs, dbg).values()
            return builder.op_multiply(lhs, rhs, dbg)
        elif (isinstance(lhs, ListValue) or isinstance(lhs, TupleValue)) and isinstance(rhs, NDArrayValue):
            lhs, rhs = builder.op_implicit_type_align(lhs, rhs, dbg).values()
            return builder.op_multiply(lhs, rhs, dbg)
        return super().build(builder, kwargs, dbg)
