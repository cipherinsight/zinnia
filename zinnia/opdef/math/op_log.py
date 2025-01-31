from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import IntegerValue, FloatValue, NDArrayValue, Value


class Math_LogOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "math.log"

    @classmethod
    def get_name(cls) -> str:
        return "log"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_log_f(builder.ir_float_cast(x), dbg=dbg)
        elif isinstance(x, FloatValue):
            return builder.ir_log_f(x, dbg=dbg)
        elif isinstance(x, NDArrayValue) and x.dtype() == IntegerType:
            return x.unary(IntegerType, lambda v: builder.ir_log_f(builder.ir_float_cast(v), dbg=dbg))
        elif isinstance(x, NDArrayValue) and x.dtype() == FloatType:
            return x.unary(IntegerType, lambda v: builder.ir_log_f(v, dbg=dbg))
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
