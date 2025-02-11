from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_ndarray_item_slice import AbstractNDArrayItemSlice
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, NumberValue, ListValue, TupleValue, IntegerValue, \
    FloatValue
from zinnia.op_def.internal.op_implicit_type_cast import ImplicitTypeCastOp


class NDArray_SetItemOp(AbstractNDArrayItemSlice):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.__set_item__"

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

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs['self']
        the_value = kwargs['value']
        statement_cond = kwargs.get_condition()
        slicing_params = self.check_slicing_params_datatype(kwargs['slicing_params'], dbg)
        assert isinstance(the_self, NDArrayValue)
        self.check_slicing_dimensions(slicing_params.values(), the_self.shape(), dbg)
        candidates, conditions = self.find_all_candidates(builder, slicing_params.values(), the_self.shape(), dbg)
        new_ndarray = the_self
        for candidate, condition in zip(candidates, conditions):
            original_value = new_ndarray.get_item(candidate)
            if isinstance(original_value, NumberValue):
                processed_value = the_value
                if ImplicitTypeCastOp.verify_cast_ability(processed_value.type(), original_value.type()):
                    processed_value = builder.op_implicit_type_cast(processed_value, original_value.type(), dbg)
                elif isinstance(processed_value, FloatValue) and isinstance(original_value, IntegerValue):
                    # TODO: raise a warning here
                    processed_value = builder.ir_int_cast(processed_value, dbg)
                elif processed_value.type() != original_value.type():
                    raise TypeInferenceError(dbg, f"Cannot assign {the_value.type()} to {original_value.type()}")
                new_value = builder.op_select(builder.ir_logical_and(statement_cond, condition), processed_value, original_value)
                new_ndarray = new_ndarray.set_item(candidate, new_value)
            elif isinstance(original_value, NDArrayValue):
                _value_ary = the_value
                if isinstance(the_value, ListValue) or isinstance(the_value, TupleValue):
                    _value_ary = builder.op_np_asarray(the_value, dbg)
                if ImplicitTypeCastOp.verify_cast_ability(the_value.type(), original_value.type()):
                    _value_ary = builder.op_implicit_type_cast(the_value, original_value.type(), dbg)
                if isinstance(the_value, NumberValue):
                    _value_ary = NDArrayValue.from_number(the_value)
                if not isinstance(_value_ary, NDArrayValue):
                    raise TypeInferenceError(dbg, f"Expected NDArray or a number, got {the_value.type()}")
                if not _value_ary.broadcast_to_compatible(original_value.shape()):
                    raise TypeInferenceError(dbg, f"Cannot broadcast input array from {_value_ary.shape()} to {original_value.shape}")
                if _value_ary.dtype() != original_value.dtype():
                    # TODO: raise a warning if casting from float to int
                    _value_ary = builder.op_ndarray_astype(_value_ary, builder.op_constant_class(original_value.dtype()))
                _value_ary = _value_ary.broadcast_to(original_value.shape())
                new_value = builder.op_select(builder.ir_logical_and(statement_cond, condition), _value_ary, original_value)
                new_ndarray = new_ndarray.set_item(candidate, new_value)
        the_self.assign(new_ndarray)
        return the_value
