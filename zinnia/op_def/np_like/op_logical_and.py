from typing import Dict, Optional, List

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_op import AbstractOp


class NP_LogicalAndOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.logical_and"

    @classmethod
    def get_name(cls) -> str:
        return "logical_and"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x1"),
            AbstractOp._ParamEntry("x2"),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        x1, x2 = kwargs["x1"], kwargs["x2"]
        return builder.op_logical_and(x1, x2, dbg)
