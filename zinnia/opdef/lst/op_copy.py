import copy
from typing import List, Dict, Optional

from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, ListValue


class List_CopyOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.copy"

    @classmethod
    def get_name(cls) -> str:
        return "copy"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, ListValue)
        return copy.copy(the_self)
