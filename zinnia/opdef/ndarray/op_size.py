from typing import List, Dict, Optional

from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NDArrayValue


class NDArray_SizeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.size"

    @classmethod
    def get_name(cls) -> str:
        return "size"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, NDArrayValue)
        number_elements = 1
        for i in the_self.shape():
            number_elements *= i
        return builder.ir_constant_int(number_elements)
