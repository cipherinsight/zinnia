from typing import Callable, Optional, Dict

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.abstract.abstract_arithemetic import AbstractArithemetic
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue, IntegerValue, FloatValue, NDArrayValue, ListValue, TupleValue, \
    Value


class PowerOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "power"

    @classmethod
    def get_name(cls) -> str:
        return "power"

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return builder.ir_pow_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                if lhs.val() is not None and lhs.val() < 0:
                    raise TypeInferenceError(None, "Math domain error. Complex values are not supported yet.")
                builder.op_assert(builder.op_logical_not(builder.op_less_than(lhs, builder.ir_constant_float(0))), builder.op_constant_none(), dbg=None)
                return builder.ir_pow_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                if lhs.val() is not None and lhs.val() < 0:
                    raise TypeInferenceError(None, "Math domain error. Complex values are not supported yet.")
                builder.op_assert(builder.op_logical_not(builder.op_less_than(lhs, builder.ir_constant_int(0))), builder.op_constant_none(), dbg=None)
                return builder.ir_pow_f(builder.ir_float_cast(lhs), rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                if lhs.val() is not None and lhs.val() < 0:
                    raise TypeInferenceError(None, "Math domain error. Complex values are not supported yet.")
                builder.op_assert(builder.op_logical_not(builder.op_less_than(lhs, builder.ir_constant_float(0))), builder.op_constant_none(), dbg=None)
                return builder.ir_pow_f(lhs, builder.ir_float_cast(rhs))
            raise NotImplementedError()
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
