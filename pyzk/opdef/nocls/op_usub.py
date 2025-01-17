from typing import List, Dict, Optional

from pyzk.debug.dbg_info import DebugInfo
from pyzk.debug.exception import TypeInferenceError
from pyzk.internal.dt_descriptor import IntegerType, FloatType
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, IntegerValue, FloatValue, NDArrayValue


class USubOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "usub"

    @classmethod
    def get_name(cls) -> str:
        return "usub"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return reducer.ir_sub_i(reducer.ir_constant_int(0), x)
        elif isinstance(x, FloatValue):
            return reducer.ir_sub_f(reducer.ir_constant_float(0.0), x)
        elif isinstance(x, NDArrayValue) and x.dtype() == IntegerType:
            return x.unary(IntegerType, lambda v: reducer.ir_sub_i(reducer.ir_constant_int(0), v))
        elif isinstance(x, NDArrayValue) and x.dtype() == FloatType:
            return x.unary(FloatType, lambda v: reducer.ir_sub_f(reducer.ir_constant_float(0.0), v))
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
