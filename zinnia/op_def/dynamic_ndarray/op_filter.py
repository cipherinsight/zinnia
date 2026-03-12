from typing import List, Optional, cast

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, IntegerValue, NDArrayValue, NumberValue, Value
from zinnia.compile.type_sys import BooleanType, FloatType, IntegerType, NumberDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.op_def.dynamic_ndarray.view_utils import default_like


class DynamicNDArray_FilterOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "DynamicNDArray.filter"

    @classmethod
    def get_name(cls) -> str:
        return "filter"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("mask"),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        mask = kwargs["mask"]
        if not isinstance(the_self, DynamicNDArrayValue):
            raise TypeInferenceError(dbg, "Param `self` must be DynamicNDArray")
        if not isinstance(mask, NDArrayValue):
            raise TypeInferenceError(dbg, "Param `mask` must be NDArray")

        src_values = the_self.flattened_values()
        mask_values = mask.flattened_values()
        max_len = the_self.max_length()

        # For dynamic masks, respect runtime logical length instead of forcing
        # bounded payload size equality with the filtered array.
        if isinstance(mask, DynamicNDArrayValue):
            mask_runtime_len = mask.runtime_logical_length()
            runtime_len_const = mask_runtime_len.val(builder)
            if runtime_len_const is not None:
                mask_runtime_len = builder.ir_constant_int(runtime_len_const)
        else:
            mask_runtime_len = builder.ir_constant_int(len(mask_values))

        mask_len_non_negative = builder.op_greater_than_or_equal(mask_runtime_len, builder.ir_constant_int(0))
        builder.op_assert(mask_len_non_negative, builder.op_constant_none(), dbg)

        mask_len_in_payload = builder.op_less_than_or_equal(mask_runtime_len, builder.ir_constant_int(len(mask_values)))
        builder.op_assert(mask_len_in_payload, builder.op_constant_none(), dbg)

        fallback = default_like(builder, cast(NumberValue, src_values[0])) if len(src_values) > 0 else builder.ir_constant_int(0)
        out_values: List[NumberValue] = [fallback for _ in range(max_len)]
        write_ptr = builder.ir_constant_int(0)
        for i in range(max_len):
            src_val = src_values[i] if i < len(src_values) else fallback
            mask_i = mask_values[i] if i < len(mask_values) else builder.ir_constant_int(0)
            mask_i_active = builder.op_less_than(builder.ir_constant_int(i), mask_runtime_len)
            if mask.dtype() == BooleanType:
                mask_is_set_raw = builder.op_bool_cast(mask_i)
            elif mask.dtype() == IntegerType:
                mask_is_set_raw = builder.op_bool_cast(builder.op_not_equal(mask_i, builder.ir_constant_int(0)))
            elif mask.dtype() == FloatType:
                mask_is_set_raw = builder.op_bool_cast(builder.op_not_equal(mask_i, builder.ir_constant_float(0.0)))
            else:
                raise TypeInferenceError(dbg, "Mask dtype must be bool/int/float")
            mask_is_set = builder.op_logical_and(mask_i_active, mask_is_set_raw)

            for j in range(max_len):
                ptr_is_j = builder.op_equal(write_ptr, builder.ir_constant_int(j))
                should_write_j = builder.op_logical_and(mask_is_set, builder.op_bool_cast(ptr_is_j))
                out_values[j] = cast(NumberValue, builder.op_select(builder.op_bool_cast(should_write_j), src_val, out_values[j]))

            next_ptr = cast(IntegerValue, builder.op_add(write_ptr, builder.ir_constant_int(1)))
            write_ptr = builder.ir_select_i(mask_is_set, next_ptr, write_ptr)

        return DynamicNDArrayValue.from_max_bounds_and_vector(
            max_len,
            the_self.max_rank(),
            cast(NumberDTDescriptor, the_self.dtype()),
            out_values,
            logical_shape=(max_len,),
            logical_offset=0,
            logical_strides=(1,),
            runtime_logical_length=write_ptr,
        )
