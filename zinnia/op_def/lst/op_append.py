from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, ListValue, NoneValue


class List_AppendOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.append"

    @classmethod
    def get_name(cls) -> str:
        return "append"

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
        value = kwargs["value"]
        assert isinstance(the_self, ListValue)
        if the_self.type_locked():
            raise TypeInferenceError(dbg, f"Cannot perform append, as it modifies the datatype on the list which is defined at parent scope.")
        new_value = ListValue(
            the_self.types() + [value.type()],
            the_self.values() + [value]
        )
        the_self.assign(new_value)
        return NoneValue()
