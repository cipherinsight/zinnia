import copy
from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.compile.type_sys import FloatType, IntegerType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import IntegerValue, FloatValue, Value, NDArrayValue


class IntCastOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "int"

    @classmethod
    def get_name(cls) -> str:
        return "int"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, FloatValue):
            return builder.ir_int_cast(x)
        elif isinstance(x, IntegerValue):
            return copy.copy(x)
        elif isinstance(x, NDArrayValue):
            if x.dtype() == FloatType:
                return x.unary(IntegerType, lambda u: builder.ir_int_cast(u))
            elif x.dtype() == IntegerType:
                return x.unary(IntegerType, lambda u: copy.copy(u))
        raise TypeInferenceError(dbg, f'Int cast on `{x.type()}` is not defined')
