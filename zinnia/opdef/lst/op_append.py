from typing import List, Dict, Optional

from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, ListValue, NoneValue


class List_AppendOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.append"

    @classmethod
    def get_name(cls) -> str:
        return "append"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("value")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        value = kwargs["value"]
        assert isinstance(the_self, ListValue)
        new_value = ListValue(
            the_self.types() + [value.type()],
            the_self.values() + [value]
        )
        the_self.assign(new_value)
        return NoneValue()
