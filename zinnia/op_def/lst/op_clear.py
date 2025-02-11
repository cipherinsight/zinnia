from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, ListValue, NoneValue


class List_ClearOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.clear"

    @classmethod
    def get_name(cls) -> str:
        return "clear"

    @classmethod
    def is_inplace(cls) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, ListValue)
        new_value = ListValue([], [])
        the_self.assign(new_value)
        return NoneValue()
