from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NDArrayValue, TupleValue, ListValue


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

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        operand = kwargs["operand"]
        if isinstance(operand, NDArrayValue):
            return builder.ir_constant_int(len(operand.shape()))
        elif isinstance(operand, TupleValue):
            return builder.ir_constant_int(len(operand.types()))
        elif isinstance(operand, ListValue):
            return builder.ir_constant_int(len(operand.types()))
        raise TypeInferenceError(dbg, f'`len` on `{operand.type()}` is not defined')
