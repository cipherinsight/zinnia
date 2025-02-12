from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, ListValue, IntegerValue


class List_InsertOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "List.insert"

    @classmethod
    def get_name(cls) -> str:
        return "insert"

    @classmethod
    def is_inplace(cls) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("index"),
            AbstractOp._ParamEntry("object")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        the_index = kwargs["index"]
        the_object = kwargs["object"]
        assert isinstance(the_self, ListValue)
        if the_self.type_locked():
            raise TypeInferenceError(dbg, f"Cannot perform insert, as it modifies the datatype on the list which is defined at parent scope.")
        if not isinstance(the_index, IntegerValue):
            raise TypeInferenceError(dbg, f"Expected an integer index for `{self.get_name()}`, got {the_index.type()}")
        if the_index.val() is None:
            if not all(t == the_self.types()[0] for t in the_self.types()):
                raise TypeInferenceError(dbg, f"`index` is not statically inferrable here. In this case, all sub-elements of the list should have the same type.")
            if not the_object.type() == the_self.types()[0]:
                raise TypeInferenceError(dbg, f"`index` is not statically inferrable here. In this case, the new object should have the same type with the objects in the list. Expected an object of type {the_self.types()[0]}, got {the_object.type()}")
            parsed_index = builder.op_select(
                builder.op_less_than(the_index, builder.ir_constant_int(0)),
                builder.op_add(builder.ir_constant_int(len(the_self.values())), the_index),
                the_index
            )
            builder.op_assert(builder.op_logical_and(
                builder.op_less_than_or_equal(builder.ir_constant_int(0), parsed_index),
                builder.op_less_than(parsed_index, builder.ir_constant_int(len(the_self.values())))
            ), builder.op_constant_none(), dbg)
            result = ListValue([the_object.type()] + the_self.types(), [the_object] + the_self.values())
            for i in range(1, len(the_self.values())):
                new_types = the_self.types().copy()
                new_types.insert(i, the_object.type())
                new_values = the_self.values().copy()
                new_values.insert(i, the_object)
                new_list = ListValue(new_types, new_values)
                result = builder.op_select(
                    builder.op_bool_cast(builder.op_equal(parsed_index, builder.ir_constant_int(i), dbg), dbg),
                    new_list, result
                )
            return result
        parsed_index = the_index.val() if the_index.val() >= 0 else len(the_self.values()) + the_index.val()
        new_types = the_self.types().copy()
        new_types.insert(parsed_index, the_object.type())
        new_values = the_self.values().copy()
        new_values.insert(parsed_index, the_object)
        new_list = ListValue(new_types, new_values)
        return new_list
