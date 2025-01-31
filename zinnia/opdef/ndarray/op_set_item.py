from typing import List, Optional, Dict

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.opdef.abstract.abstract_ndarray_item_slice import AbstractNDArrayItemSlice
from zinnia.opdef.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NDArrayValue, NumberValue, ListValue, TupleValue, IntegerValue, \
    FloatValue


class NDArray_SetItemOp(AbstractNDArrayItemSlice):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.__set_item__"

    @classmethod
    def get_name(cls) -> str:
        return "__set_item__"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("value"),
            AbstractOp._ParamEntry("slicing_params")
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs['self']
        the_value = kwargs['value']
        slicing_params = self.check_slicing_params_datatype(kwargs['slicing_params'], dbg)
        assert isinstance(the_self, NDArrayValue)
        self.check_slicing_dimensions(slicing_params.values(), the_self.shape(), dbg)
        candidates, conditions = self.find_all_candidates(builder, slicing_params.values(), the_self.shape(), dbg)
        new_ndarray = the_self
        for candidate, condition in zip(candidates, conditions):
            original_value = new_ndarray.get_item(candidate)
            if isinstance(original_value, NumberValue):
                if isinstance(the_value, IntegerValue) and isinstance(original_value, FloatValue):
                    new_value = builder.op_select(condition, builder.ir_float_cast(the_value), original_value)
                    new_ndarray = new_ndarray.set_item(candidate, new_value)
                elif isinstance(the_value, FloatValue) and isinstance(original_value, IntegerValue):
                    # TODO: raise a warning here
                    new_value = builder.op_select(condition, builder.ir_int_cast(the_value), original_value)
                    new_ndarray = new_ndarray.set_item(candidate, new_value)
                elif the_value.type() == original_value.type():
                    new_value = builder.op_select(condition, the_value, original_value)
                    new_ndarray = new_ndarray.set_item(candidate, new_value)
                else:
                    raise TypeInferenceError(dbg, f"Cannot assign {the_value.type()} to {original_value.type()}")
            elif isinstance(original_value, NDArrayValue):
                _value_ary = the_value
                if isinstance(_value_ary, ListValue) or isinstance(_value_ary, TupleValue):
                    _value_ary = builder.op_ndarray_asarray(_value_ary, dbg)
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
                new_value = builder.op_select(condition, _value_ary, original_value)
                new_ndarray = new_ndarray.set_item(candidate, new_value)
        return the_value
