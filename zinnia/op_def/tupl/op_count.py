from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, TupleValue


class Tuple_CountOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "Tuple.count"

    @classmethod
    def get_name(cls) -> str:
        return "count"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("value")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        value = kwargs["value"]
        assert isinstance(the_self, TupleValue)
        answer = builder.ir_constant_int(0)
        for i, v in enumerate(the_self.values()):
            equal = builder.op_bool_cast(builder.op_equal(v, value, dbg), dbg)
            answer = builder.op_select(equal, builder.op_add(answer, builder.ir_constant_int(1)), answer, dbg)
        return answer
