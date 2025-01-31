from typing import Dict, Optional, List

from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.opdef.abstract.abstract_op import AbstractOp


class NP_DivideOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.divide"

    @classmethod
    def get_name(cls) -> str:
        return "divide"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x1"),
            AbstractOp._ParamEntry("x2"),
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        x1, x2 = kwargs["x1"], kwargs["x2"]
        return builder.op_divide(x1, x2, dbg)
