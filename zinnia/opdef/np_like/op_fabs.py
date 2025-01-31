from typing import Dict, Optional, List

from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.opdef.abstract.abstract_op import AbstractOp


class NP_FAbsOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.fabs"

    @classmethod
    def get_name(cls) -> str:
        return "fabs"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x = kwargs["x"]
        return builder.op_abs(x, dbg)
