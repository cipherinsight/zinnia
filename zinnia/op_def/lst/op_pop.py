from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, ListValue, IntegerValue, NoneValue


class List_PopOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.pop"

    @classmethod
    def get_name(cls) -> str:
        return "pop"

    @classmethod
    def is_inplace(cls) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("index", True)
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        index = kwargs.get("index", builder.ir_constant_int(-1))
        assert isinstance(the_self, ListValue)
        if not isinstance(index, IntegerValue):
            raise TypeInferenceError(dbg, f"Expected an integer index for `{self.get_name()}`, got {index.type()}")
        if the_self.type_locked():
            raise TypeInferenceError(dbg, f"Cannot perform pop, as it modifies the datatype on the list which is defined at parent scope.")
        if index.val(builder) is None:
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
            ), builder.op_constant_none(), dbg)
            result = ListValue(the_self.types()[1:], the_self.values()[1:])
            for i in range(1, len(the_self.values())):
                result = builder.op_select(
                    builder.op_bool_cast(builder.op_equal(parsed_index, builder.ir_constant_int(i), dbg), dbg),
                    ListValue(the_self.types()[:i] + the_self.types()[i + 1:], the_self.values()[:i] + the_self.values()[i + 1:]),
                    result
                )
            the_self.assign(result)
            return NoneValue()
        parsed_index = index.val(builder) if index.val(builder) >= 0 else len(the_self.values()) + index.val(builder)
        if parsed_index < 0 or parsed_index >= len(the_self.values()):
            raise TypeInferenceError(dbg, f"pop index out of range")
        new_list = ListValue(
            the_self.types()[:parsed_index] + the_self.types()[parsed_index + 1:],
            the_self.values()[:parsed_index] + the_self.values()[parsed_index + 1:]
        )
        the_self.assign(new_list)
        return NoneValue()
