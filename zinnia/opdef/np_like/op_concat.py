from typing import Optional, List, Dict

from zinnia.debug.dbg_info import DebugInfo
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value
from zinnia.opdef.np_like.op_concatenate import NP_ConcatenateOp


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

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        return super().build(builder, kwargs, dbg)
