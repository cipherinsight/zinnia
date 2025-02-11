from typing import Optional, List, Dict

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value
from zinnia.op_def.np_like.op_concatenate import NP_ConcatenateOp


class NP_ConcatOp(NP_ConcatenateOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.concat"

    @classmethod
    def get_name(cls) -> str:
        return "concat"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("arrays"),
            AbstractOp._ParamEntry("axis", default=True),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        return super().build(builder, kwargs, dbg)
