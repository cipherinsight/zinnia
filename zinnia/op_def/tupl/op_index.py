from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import StaticInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, TupleValue, NoneValue, IntegerValue


class Tuple_IndexOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "Tuple.index"

    @classmethod
    def get_name(cls) -> str:
        return "index"

    @classmethod
    def requires_condition(cls) -> bool:
        return False

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("value"),
            AbstractOp._ParamEntry("start", True),
            AbstractOp._ParamEntry("stop", True),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        value = kwargs["value"]
        assert isinstance(the_self, TupleValue)
        start = kwargs.get("start", builder.op_constant_none())
        stop = kwargs.get("stop", builder.op_constant_none())
        if isinstance(start, NoneValue):
            start = builder.ir_constant_int(0)
        if isinstance(stop, NoneValue):
            stop = builder.ir_constant_int(len(the_self.values()))
        if not isinstance(start, IntegerValue):
            raise StaticInferenceError(dbg, f"`start` must be an integer")
        if not isinstance(stop, IntegerValue):
            raise StaticInferenceError(dbg, f"`stop` must be an integer")
        found_answer = builder.ir_constant_bool(False)
        answer = builder.ir_constant_int(0)
        for i, v in enumerate(the_self.values()):
            equal = builder.op_bool_cast(builder.op_equal(v, value, dbg), dbg)
            equal = builder.ir_logical_and(equal, builder.ir_logical_and(
                builder.ir_less_than_or_equal_i(start, builder.ir_constant_int(i), dbg),
                builder.ir_less_than_i(builder.ir_constant_int(i), stop, dbg),
            ))
            answer = builder.op_select(builder.ir_logical_and(equal, builder.ir_logical_not(found_answer)), builder.ir_constant_int(i), answer, dbg)
            found_answer = builder.op_logical_or(equal, found_answer, dbg)
        if found_answer.val(builder) is not None and found_answer.val(builder) == 0:
            raise StaticInferenceError(dbg, f"Value not found in tuple")
        # builder.op_assert(found_answer, kwargs.get_condition(), dbg)
        return answer
