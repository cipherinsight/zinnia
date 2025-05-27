from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, ListValue, NoneValue


class List_ReverseOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.reverse"

    @classmethod
    def get_name(cls) -> str:
        return "reverse"

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
        if the_self.type_locked() and not all(tp == the_self.types()[0] for tp in the_self.types()):
            raise TypeInferenceError(dbg, f"Cannot perform reverse, as it modifies the datatype on the list which is defined at parent scope.")
        new_value = ListValue(
            list(reversed(the_self.types())),
            list(reversed(the_self.values())),
        )
        the_self.assign(new_value)
        return NoneValue()
