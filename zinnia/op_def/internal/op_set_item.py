from typing import List, Optional, Callable

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.op_def.abstract.abstract_item_slice import AbstractItemSliceOp
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, TupleValue, IntegerValue, ListValue, BooleanValue
from zinnia.op_def.internal.op_implicit_type_cast import ImplicitTypeCastOp


class SetItemOp(AbstractItemSliceOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "__set_item__"

    @classmethod
    def get_name(cls) -> str:
        return "__set_item__"

    @classmethod
    def is_inplace(cls) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("value"),
            AbstractOp._ParamEntry("slicing_params")
        ]

    def _build_list_assignment(
            self,
            builder: IRBuilderInterface,
            condition: BooleanValue,
            augment_op: Callable[[Value, Value], Value],
            the_self: ListValue, the_value: Value, slicing_param: TupleValue | IntegerValue,
            dbg: Optional[DebugInfo] = None
    ) -> Value:
        if isinstance(slicing_param, IntegerValue):
            if slicing_param.val() is not None:
                self.check_single_slicing_number(slicing_param, len(the_self.values()), dbg)
                the_index = slicing_param.val() if slicing_param.val() >= 0 else len(the_self.values()) + slicing_param.val()
                if the_self.type_locked():
                    if ImplicitTypeCastOp.verify_cast_ability(the_value.type(), the_self.types()[the_index]):
                        processed_value = builder.op_implicit_type_cast(the_value, the_self.types()[the_index], dbg)
                        processed_value = builder.op_select(
                            condition,
                            augment_op(the_self.values()[the_index], processed_value),
                            the_self.values()[the_index]
                        )
                        the_self.values()[the_index].assign(processed_value)
                    else:
                        raise StaticInferenceError(dbg,"Cannot change the data type of the list element, as the list is not defined at this scope")
                else:
                    new_types = the_self.types()[:the_index] + [the_value.type()] + the_self.types()[the_index + 1:]
                    new_values = the_self.values()[:the_index] + [the_value] + the_self.values()[the_index + 1:]
                    the_self.assign(ListValue(new_types, new_values))
                return the_value
            all_datatype_equal = all(x == the_self.types()[0] for x in the_self.types()[1:])
            if not all_datatype_equal:
                raise StaticInferenceError(dbg, f"{the_self.type()} set_item: all elements in the {the_self.type()} should have the same data type, otherwise the result data type is non-deterministic")
            if ImplicitTypeCastOp.verify_cast_ability(the_value.type(), the_self.types()[0]):
                processed_value = builder.op_implicit_type_cast(the_value, the_self.types()[0])
            else:
                raise StaticInferenceError(dbg, f"{the_self.type()} set_item: the value type is not equal to the element type of the list")
            self.insert_slicing_number_assertion(slicing_param, condition, len(the_self.values()), builder)
            new_values = the_self.values()
            for i, v in enumerate(the_self.values()):
                selected_value = builder.op_select(condition, processed_value, the_self.values()[i])
                new_values[i] = builder.op_select(
                    builder.ir_equal_i(slicing_param, builder.ir_constant_int(i)),
                    augment_op(v, selected_value),
                    v
                )
            the_self.assign(ListValue(the_self.types(), new_values))
            return the_value
        elif isinstance(slicing_param, TupleValue):
            [start, stop, step] = slicing_param.values()
            start = start.val() if isinstance(start, IntegerValue) else None
            stop = stop.val() if isinstance(stop, IntegerValue) else None
            step = step.val() if isinstance(step, IntegerValue) else None
            converted_value = builder.op_list_cast(the_value)
            if step is None or step == 1:
                the_self.assign(ListValue(
                    the_self.types()[:start] + converted_value.types() + the_self.types()[stop:],
                    the_self.values()[:start] + converted_value.values() + the_self.values()[stop:])
                )
                return the_value
            replace_indices = range(len(the_self.values()))[start:stop:step]
            if len(converted_value.values()) != len(replace_indices):
                raise StaticInferenceError(dbg, f"Invalid set_item: attempt to assign sequence of size {len(converted_value.values())} to extended slice of size {len(replace_indices)}")
            new_types, new_values = the_self.types(), the_self.values()
            for i, v in zip(replace_indices, converted_value.values()):
                new_types[i] = v.type()
                new_values[i] = v
            the_self.assign(ListValue(new_types, new_values))
            return the_value
        raise NotImplementedError()

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs['self']
        the_value = kwargs['value']
        slicing_params = self.check_slicing_params_datatype(kwargs['slicing_params'], dbg)
        if isinstance(the_self, ListValue):
            if len(slicing_params.values()) != 1:
                raise StaticInferenceError(dbg, f"List set_item should have exactly one slicing parameter")
            slicing_param = slicing_params.values()[0]
            return self._build_list_assignment(builder, kwargs.get_condition(), lambda x, y: y, the_self, the_value, slicing_param, dbg)
        if isinstance(the_self, NDArrayValue):
            return builder.op_ndarray_set_item(kwargs.get_condition(), the_self, slicing_params, the_value, dbg)
        raise TypeInferenceError(dbg, f"{the_self.type()} does not support item assignment")
