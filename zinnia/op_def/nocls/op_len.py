from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, TupleValue, ListValue


class LenOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "len"

    @classmethod
    def get_name(cls) -> str:
        return "len"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("operand")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        operand = kwargs["operand"]
        if isinstance(operand, NDArrayValue):
            return builder.ir_constant_int(operand.shape()[0])
        elif isinstance(operand, TupleValue):
            return builder.ir_constant_int(len(operand.types()))
        elif isinstance(operand, ListValue):
            return builder.ir_constant_int(len(operand.types()))
        raise TypeInferenceError(dbg, f'`len` on `{operand.type()}` is not defined')
