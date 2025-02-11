from typing import Callable

from zinnia.op_def.abstract.abstract_arithemetic import AbstractArithemetic
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import IntegerValue, NumberValue, FloatValue


class NP_MinimumOp(AbstractArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.minimum"

    @classmethod
    def get_name(cls) -> str:
        return "minimum"

    def get_build_op_lambda(self, builder: IRBuilderInterface) -> Callable[[NumberValue, NumberValue], NumberValue]:
        def _inner(lhs: NumberValue, rhs: NumberValue) -> NumberValue:
            if isinstance(lhs, IntegerValue) and isinstance(rhs, IntegerValue):
                return builder.ir_select_i(builder.ir_bool_cast(builder.ir_less_than_i(lhs, rhs)), lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, FloatValue):
                return builder.ir_select_f(builder.ir_bool_cast(builder.ir_less_than_f(lhs, rhs)), lhs, rhs)
            elif isinstance(lhs, IntegerValue) and isinstance(rhs, FloatValue):
                lhs = builder.ir_float_cast(lhs)
                return builder.ir_select_f(builder.ir_bool_cast(builder.ir_less_than_f(lhs, rhs)), lhs, rhs)
            elif isinstance(lhs, FloatValue) and isinstance(rhs, IntegerValue):
                rhs = builder.ir_float_cast(rhs)
                return builder.ir_select_f(builder.ir_bool_cast(builder.ir_less_than_f(lhs, rhs)), lhs, rhs)
            raise NotImplementedError()
        return _inner
