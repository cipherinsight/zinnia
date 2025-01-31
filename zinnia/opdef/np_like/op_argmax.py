from typing import Dict, Optional, List

from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.opdef.abstract.abstract_op import AbstractOp


class NP_ArgmaxOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.argmax"

    @classmethod
    def get_name(cls) -> str:
        return "argmax"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("a"),
            AbstractOp._ParamEntry("axis", True),
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        a, axis = kwargs["a"], kwargs.get("axis", None)
        return builder.op_ndarray_argmax(a, axis, dbg)
