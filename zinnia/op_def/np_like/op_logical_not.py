from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.type_sys import IntegerType, BooleanType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue, NDArrayValue, BooleanValue


class NP_LogicalNotOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.logical_not"

    @classmethod
    def get_name(cls) -> str:
        return "logical_not"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, BooleanValue):
            return builder.ir_logical_not(x)
        elif isinstance(x, IntegerValue):
            return builder.ir_logical_not(builder.op_bool_cast(x))
        elif isinstance(x, NDArrayValue) and x.dtype() == BooleanType:
            return x.unary(BooleanType, lambda v: builder.ir_logical_not(v))
        elif isinstance(x, NDArrayValue) and x.dtype() == IntegerType:
            return x.unary(BooleanType, lambda v: builder.ir_logical_not(builder.op_bool_cast(v)))
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
