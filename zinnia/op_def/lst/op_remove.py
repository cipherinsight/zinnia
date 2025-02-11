from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, ListValue, NoneValue


class List_RemoveOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.remove"

    @classmethod
    def get_name(cls) -> str:
        return "remove"

    @classmethod
    def is_inplace(cls) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("value")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        the_value = kwargs["value"]
        assert isinstance(the_self, ListValue)
        the_index = builder.op_list_index(the_self, the_value, builder.op_constant_none(), builder.op_constant_none(), dbg)
        builder.op_list_pop(kwargs.get_condition(), the_self, the_index, dbg)
        return NoneValue()
