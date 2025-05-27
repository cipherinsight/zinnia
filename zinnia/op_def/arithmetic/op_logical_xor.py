from typing import Callable

from zinnia.op_def.abstract.abstract_integer_arithemetic import AbstractLogicalArithemetic
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import IntegerValue


class LogicalXorOp(AbstractLogicalArithemetic):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "logical_xor"

    @classmethod
    def get_name(cls) -> str:
        return "logical_xor"

    def get_build_op_lambda(self, builder: IRBuilderInterface) -> Callable[[IntegerValue, IntegerValue], IntegerValue]:
        return lambda x, y: builder.ir_not_equal_i(builder.ir_bool_cast(x), builder.ir_bool_cast(y))
