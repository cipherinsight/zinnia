from typing import Callable

from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import NumberValue, IntegerValue, FloatValue
from zinnia.opdef.abstract.abstract_arithemetic import AbstractArithemetic


class NP_ModOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.mod"

    @classmethod
    def get_name(cls) -> str:
        return "mod"

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return builder.ir_mod_i(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return builder.ir_mod_f(lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                lhs = builder.ir_float_cast(lhs)
                return builder.ir_mod_f(lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                rhs = builder.ir_float_cast(rhs)
                return builder.ir_mod_f(lhs, rhs)
            raise NotImplementedError()
        return _inner
