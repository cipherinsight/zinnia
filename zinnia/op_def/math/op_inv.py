from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue


class Math_InvOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "math.inv"

    @classmethod
    def get_name(cls) -> str:
        return "inv"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        if isinstance(x, IntegerValue):
            return builder.ir_inv_i(x, dbg)
        raise TypeInferenceError(dbg, f"Unsupported argument type for `{self.get_name()}`: {x.type()}")
