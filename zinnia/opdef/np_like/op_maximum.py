from typing import Callable

from zinnia.opdef.abstract.abstract_arithemetic import AbstractArithemetic
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import FloatValue, IntegerValue, NumberValue


class NP_MaximumOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.maximum"

    @classmethod
    def get_name(cls) -> str:
        return "maximum"

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return builder.ir_select_i(builder.ir_bool_cast(builder.ir_less_than_i(lhs, rhs)), rhs, lhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return builder.ir_select_f(builder.ir_bool_cast(builder.ir_less_than_f(lhs, rhs)), rhs, lhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                lhs = builder.ir_float_cast(lhs)
                return builder.ir_select_f(builder.ir_bool_cast(builder.ir_less_than_f(lhs, rhs)), rhs, lhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                rhs = builder.ir_float_cast(rhs)
                return builder.ir_select_f(builder.ir_bool_cast(builder.ir_less_than_f(lhs, rhs)), rhs, lhs)
            raise NotImplementedError()
        return _inner
