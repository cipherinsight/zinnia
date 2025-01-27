import copy
from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import FloatType, IntegerType
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, IntegerValue, FloatValue, NDArrayValue


class FloatCastOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "float"

    @classmethod
    def get_name(cls) -> str:
        return "float"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_float_cast(x)
        elif isinstance(x, FloatValue):
            return copy.copy(x)
        elif isinstance(x, NDArrayValue):
            if x.dtype() == FloatType:
                return x.unary(FloatType, lambda u: copy.copy(u))
            elif x.dtype() == IntegerType:
                return x.unary(FloatType, lambda u: builder.ir_float_cast(u))
        raise TypeInferenceError(dbg, f'Float cast on `{x.type()}` is not defined')
