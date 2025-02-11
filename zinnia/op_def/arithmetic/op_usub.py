from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue, FloatValue, NDArrayValue


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

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_sub_i(builder.ir_constant_int(0), x)
        elif isinstance(x, FloatValue):
            return builder.ir_sub_f(builder.ir_constant_float(0.0), x)
        elif isinstance(x, NDArrayValue) and x.dtype() == IntegerType:
            return x.unary(IntegerType, lambda v: builder.ir_sub_i(builder.ir_constant_int(0), v))
        elif isinstance(x, NDArrayValue) and x.dtype() == FloatType:
            return x.unary(FloatType, lambda v: builder.ir_sub_f(builder.ir_constant_float(0.0), v))
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
