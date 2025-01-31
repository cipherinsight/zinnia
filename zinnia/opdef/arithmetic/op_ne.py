from typing import Callable, Optional, Dict

from zinnia.debug.dbg_info import DebugInfo
from zinnia.opdef.abstract.abstract_compare import AbstractCompare
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue, IntegerValue, FloatValue, Value, TupleValue, ListValue, \
    ClassValue, NDArrayValue


class NotEqualOp(AbstractCompare):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "ne"

    @classmethod
    def get_name(cls) -> str:
        return "ne"

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return builder.ir_not_equal_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return builder.ir_not_equal_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                return builder.ir_not_equal_f(builder.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                return builder.ir_not_equal_f(lhs, builder.ir_float_cast(rhs))
            raise NotImplementedError()
        return _inner

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, TupleValue) and isinstance(rhs, TupleValue):
            if lhs.types() != rhs.types():
                return builder.ir_constant_int(1)
            result = builder.ir_constant_int(0)
            for l, r in zip(lhs.values(), rhs.values()):
                result = builder.ir_logical_or(result, builder.op_bool_cast(builder.op_not_equal(l, r)))
            return result
        elif isinstance(lhs, ListValue) and isinstance(rhs, ListValue):
            if lhs.types() != rhs.types():
                return builder.ir_constant_int(1)
            result = builder.ir_constant_int(0)
            for l, r in zip(lhs.values(), rhs.values()):
                result = builder.ir_logical_or(result, builder.op_bool_cast(builder.op_not_equal(l, r)))
            return result
        elif isinstance(lhs, ClassValue) and isinstance(rhs, ClassValue):
            return builder.ir_constant_int(1) if lhs.val() != rhs.val() else builder.ir_constant_int(0)
        elif isinstance(lhs, ClassValue) and isinstance(rhs, ClassValue):
            return builder.ir_constant_int(1) if lhs.val() == rhs.val() else builder.ir_constant_int(0)
        elif isinstance(lhs, NDArrayValue) and (isinstance(rhs, ListValue) or isinstance(rhs, TupleValue)):
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
