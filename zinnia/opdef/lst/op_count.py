from typing import List, Dict, Optional

from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, ListValue


class List_CountOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.count"

    @classmethod
    def get_name(cls) -> str:
        return "count"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("value")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        value = kwargs["value"]
        assert isinstance(the_self, ListValue)
        answer = builder.ir_constant_int(0)
        for i, v in enumerate(the_self.values()):
            equal = builder.op_bool_cast(builder.op_equal(v, value, dbg), dbg)
            answer = builder.op_select(equal, builder.op_add(answer, builder.ir_constant_int(1)), answer, dbg)
        return answer
