from typing import List, Dict, Optional

from zinnia.debug.exception import StaticInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, ListValue, NoneValue


class List_IndexOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.index"

    @classmethod
    def get_name(cls) -> str:
        return "index"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("value")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        value = kwargs["value"]
        assert isinstance(the_self, ListValue)
        found_answer = builder.ir_constant_int(0)
        answer = builder.ir_constant_int(0)
        for i, v in the_self.values():
            equal = builder.op_bool_cast(builder.op_equal(v, value, dbg), dbg)
            answer = builder.op_select(builder.ir_logical_and(equal, builder.ir_logical_not(found_answer)), builder.ir_constant_int(i), answer, dbg)
            found_answer = builder.op_select(equal, builder.ir_constant_int(1), found_answer, dbg)
        if found_answer.val() is not None and found_answer.val() == 0:
            raise StaticInferenceError(dbg, f"Value not found in list")
        builder.op_assert(found_answer, builder.op_constant_none(), dbg)
        return answer
