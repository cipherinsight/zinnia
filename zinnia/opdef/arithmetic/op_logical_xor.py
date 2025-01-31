from typing import Callable

from zinnia.opdef.abstract.abstract_integer_arithemetic import AbstractIntegerArithemetic
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import IntegerValue


class LogicalXorOp(AbstractIntegerArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_xor"

    @classmethod
    def get_name(cls) -> str:
        return "logical_xor"

    def get_build_op_lambda(self, builder: AbsIRBuilderInterface) -> Callable[[IntegerValue, IntegerValue], IntegerValue]:
        return lambda x, y: builder.ir_not_equal_i(builder.ir_bool_cast(x), builder.ir_bool_cast(y))
