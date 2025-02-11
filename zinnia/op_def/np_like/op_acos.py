from typing import Dict, Optional, List

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_op import AbstractOp


class NP_ACosOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.acos"

    @classmethod
    def get_name(cls) -> str:
        return "acos"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        raise NotImplementedError()
