import copy
from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.internal.dt_descriptor import FloatType, IntegerType
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.debug.dbg_info import DebugInfo
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, IntegerValue, FloatValue, NDArrayValue


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

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return reducer.ir_float_cast(x)
        elif isinstance(x, FloatValue):
            return copy.copy(x)
        elif isinstance(x, NDArrayValue):
            if x.dtype() == FloatType:
                return x.unary(FloatType, lambda u: copy.copy(u))
            elif x.dtype() == IntegerType:
                return x.unary(FloatType, lambda u: reducer.ir_float_cast(u))
        raise TypeInferenceError(dbg, f'Float cast on `{x.type()}` is not defined')
