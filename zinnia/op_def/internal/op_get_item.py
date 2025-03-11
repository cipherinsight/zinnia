from typing import Dict, List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.op_def.abstract.abstract_item_slice import AbstractItemSliceOp
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, ListValue, TupleValue, IntegerValue, NDArrayValue, BooleanValue


class GetItemOp(AbstractItemSliceOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "__get_item__"

    @classmethod
    def get_name(cls) -> str:
        return "__get_item__"

    @classmethod
    def requires_condition(cls) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("slicing_params")
        ]

    def _build_tuple_list_slicing(self, builder: IRBuilderInterface, the_self: TupleValue | ListValue, slicing_param: TupleValue | IntegerValue, condition: BooleanValue, dbg: Optional[DebugInfo] = None) -> Value:
        if isinstance(slicing_param, IntegerValue):
            if slicing_param.val() is not None:
                self.check_single_slicing_number(slicing_param, len(the_self.values()), dbg)
                return the_self.values()[slicing_param.val()]
            all_datatype_equal = all(x == the_self.types()[0] for x in the_self.types()[1:])
            if not all_datatype_equal:
                raise StaticInferenceError(dbg, f"{the_self.type()} Slicing: all elements in the {the_self.type()} should have the same data type, otherwise the result data type is non-deterministic")
            self.insert_slicing_number_assertion(slicing_param, condition, len(the_self.values()), builder)
            result = the_self.values()[0]
            for i in range(1, len(the_self.values())):
                result = builder.op_select(builder.ir_equal_i(slicing_param, builder.ir_constant_int(i)), the_self.values()[i], result)
            return result
        elif isinstance(slicing_param, TupleValue):
            [start, stop, step] = slicing_param.values()
            start = start.val() if isinstance(start, IntegerValue) else None
            stop = stop.val() if isinstance(stop, IntegerValue) else None
            step = step.val() if isinstance(step, IntegerValue) else None
            if isinstance(the_self, TupleValue):
                return TupleValue(the_self.types()[start:stop:step], the_self.values()[start:stop:step])
            elif isinstance(the_self, ListValue):
                return ListValue(the_self.types()[start:stop:step], the_self.values()[start:stop:step])
            raise NotImplementedError()

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs['self']
        slicing_params = self.check_slicing_params_datatype(kwargs['slicing_params'], dbg)
        if isinstance(the_self, TupleValue):
            if len(slicing_params.values()) != 1:
                raise StaticInferenceError(dbg, f"Tuple slicing should have exactly one slicing parameter")
            slicing_param = slicing_params.values()[0]
            return self._build_tuple_list_slicing(builder, the_self, slicing_param, kwargs.get_condition(), dbg)
        elif isinstance(the_self, ListValue):
            if len(slicing_params.values()) != 1:
                raise StaticInferenceError(dbg, f"List slicing should have exactly one slicing parameter")
            slicing_param = slicing_params.values()[0]
            return self._build_tuple_list_slicing(builder, the_self, slicing_param, kwargs.get_condition(), dbg)
        elif isinstance(the_self, NDArrayValue):
            return builder.op_ndarray_get_item(kwargs.get_condition(), the_self, slicing_params, dbg)
        raise TypeInferenceError(dbg, f"Operator `{self.get_signature()}` not defined on type {the_self.type()}")
