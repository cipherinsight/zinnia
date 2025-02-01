from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, ListValue, IntegerValue


class List_PopOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.pop"

    @classmethod
    def get_name(cls) -> str:
        return "pop"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("index", True)
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        index = kwargs.get("index", builder.ir_constant_int(-1))
        assert isinstance(the_self, ListValue)
        if not isinstance(index, IntegerValue):
            raise TypeInferenceError(dbg, f"Expected an integer index for `{self.get_name()}`, got {index.type()}")
        if index.val() is None:
            if not all(t == the_self.types()[0] for t in the_self.types()):
                raise TypeInferenceError(dbg, f"`index` is not statically inferrable here. In this case, all sub-elements of the list should have the same type.")
            parsed_index = builder.op_select(
                builder.op_less_than(index, builder.ir_constant_int(0)),
                builder.op_add(builder.ir_constant_int(len(the_self.values())), index),
                index
            )
            builder.op_assert(builder.op_logical_and(
                builder.op_less_than_or_equal(builder.ir_constant_int(0), parsed_index),
                builder.op_less_than(parsed_index, builder.ir_constant_int(len(the_self.values())))
            ), None, dbg)
            result = ListValue(the_self.types()[1:], the_self.values()[1:])
            for i in range(1, len(the_self.values())):
                result = builder.op_select(
                    builder.op_bool_cast(builder.op_equal(parsed_index, builder.ir_constant_int(i), dbg), dbg),
                    ListValue(the_self.types()[:i] + the_self.types()[i + 1:], the_self.values()[:i] + the_self.values()[i + 1:]),
                    result
                )
            return result
        parsed_index = index.val() if index.val() >= 0 else len(the_self.values()) + index.val()
        return ListValue(
            the_self.types()[:parsed_index] + the_self.types()[parsed_index + 1:],
            the_self.values()[:parsed_index] + the_self.values()[parsed_index + 1:]
        )
