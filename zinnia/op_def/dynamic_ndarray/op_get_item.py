from typing import List, Optional, cast

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import (
    DynamicNDArrayValue,
    IntegerValue,
    ListValue,
    NoneValue,
    NumberValue,
    TupleValue,
    Value,
)
from zinnia.compile.type_sys import BooleanType, FloatType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.op_def.dynamic_ndarray.view_utils import default_like


class DynamicNDArray_GetItemOp(AbstractOp):
    MUX_THRESHOLD = 100

    def get_signature(self) -> str:
        return "DynamicNDArray.__get_item__"

    @classmethod
    def get_name(cls) -> str:
        return "__get_item__"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("slicing_params"),
        ]

    def _write_source_memory(
        self,
        builder: IRBuilderInterface,
        values: List[NumberValue],
        max_len: int,
        fallback: IntegerValue,
    ) -> int:
        segment_id = len(getattr(builder, "stmts"))
        builder.ir_allocate_memory(segment_id=segment_id, size=max_len, init_value=0)
        for i in range(max_len):
            src = values[i] if i < len(values) else fallback
            builder.ir_write_memory(segment_id, builder.ir_constant_int(i), builder.op_int_cast(src))
        return segment_id

    def _select_by_index_mux(
        self,
        builder: IRBuilderInterface,
        index: IntegerValue,
        values: List[NumberValue],
        max_len: int,
        fallback: NumberValue,
        dbg: Optional[DebugInfo],
    ) -> NumberValue:
        cells = [values[i] if i < len(values) else fallback for i in range(max_len)]
        selected = cells[0]
        for i in range(1, max_len):
            cond = builder.op_equal(index, builder.ir_constant_int(i), dbg)
            selected = cast(NumberValue, builder.op_select(builder.op_bool_cast(cond, dbg), cells[i], selected, dbg))
        return selected

    def _materialize_dtype(self, builder: IRBuilderInterface, x: IntegerValue, the_self: DynamicNDArrayValue) -> Value:
        dtype = the_self.dtype()
        if dtype == FloatType:
            return builder.op_float_cast(x)
        if dtype == BooleanType:
            return builder.op_bool_cast(x)
        return x

    def _normalize_index(self, builder: IRBuilderInterface, index: IntegerValue, max_len: int, dbg: Optional[DebugInfo]) -> IntegerValue:
        is_negative = builder.op_less_than(index, builder.ir_constant_int(0), dbg)
        shifted = cast(IntegerValue, builder.op_add(index, builder.ir_constant_int(max_len), dbg))
        normalized = cast(IntegerValue, builder.op_select(builder.op_bool_cast(is_negative, dbg), shifted, index, dbg))
        index_in_range = builder.op_logical_and(
            builder.op_greater_than_or_equal(normalized, builder.ir_constant_int(0), dbg),
            builder.op_less_than(normalized, builder.ir_constant_int(max_len), dbg),
            dbg,
        )
        # builder.op_assert(index_in_range, builder.op_constant_none(), dbg)
        return normalized

    def _dynamic_scalar_index(
        self,
        builder: IRBuilderInterface,
        the_self: DynamicNDArrayValue,
        index: IntegerValue,
        dbg: Optional[DebugInfo],
    ) -> Value:
        max_len = the_self.max_length()
        src_values = the_self.flattened_values()
        normalized = self._normalize_index(builder, index, max_len, dbg)

        if max_len < self.MUX_THRESHOLD:
            fallback_num = default_like(builder, cast(NumberValue, src_values[0])) if len(src_values) > 0 else builder.ir_constant_int(0)
            return self._select_by_index_mux(builder, normalized, src_values, max_len, fallback_num, dbg)

        fallback = builder.ir_constant_int(0)
        segment_id = self._write_source_memory(builder, src_values, max_len, fallback)
        read_val = builder.ir_read_memory(segment_id, normalized)
        return self._materialize_dtype(builder, read_val, the_self)

    def _to_int_or_default(
        self,
        builder: IRBuilderInterface,
        value: Value,
        default: int,
    ) -> IntegerValue:
        if isinstance(value, NoneValue):
            return builder.ir_constant_int(default)
        if isinstance(value, IntegerValue):
            return value
        raise TypeInferenceError(None, "Slice components must be int/None")

    def _dynamic_slice(
        self,
        builder: IRBuilderInterface,
        the_self: DynamicNDArrayValue,
        slicing: TupleValue,
        dbg: Optional[DebugInfo],
    ) -> DynamicNDArrayValue:
        start_raw, stop_raw, step_raw = slicing.values()
        max_len = the_self.max_length()

        start = self._to_int_or_default(builder, start_raw, 0)
        stop = self._to_int_or_default(builder, stop_raw, max_len)
        step = self._to_int_or_default(builder, step_raw, 1)

        step_is_zero = builder.op_equal(step, builder.ir_constant_int(0), dbg)
        builder.op_assert(builder.op_logical_not(builder.op_bool_cast(step_is_zero, dbg), dbg), builder.op_constant_none(), dbg)

        is_positive_step = builder.op_greater_than(step, builder.ir_constant_int(0), dbg)
        builder.op_assert(is_positive_step, builder.op_constant_none(), dbg)

        src_values = the_self.flattened_values()
        fallback_num = default_like(builder, cast(NumberValue, src_values[0])) if len(src_values) > 0 else builder.ir_constant_int(0)
        fallback_int = builder.op_int_cast(fallback_num)
        use_mux = max_len < self.MUX_THRESHOLD
        segment_id = self._write_source_memory(builder, src_values, max_len, fallback_int) if not use_mux else -1

        out_values: List[NumberValue] = [fallback_num for _ in range(max_len)]
        write_ptr = builder.ir_constant_int(0)

        for i in range(max_len):
            i_val = builder.ir_constant_int(i)
            idx = cast(IntegerValue, builder.op_add(start, builder.op_multiply(i_val, step, dbg), dbg))
            in_low = builder.op_greater_than_or_equal(idx, builder.ir_constant_int(0), dbg)
            in_high_stop = builder.op_less_than(idx, stop, dbg)
            in_high_cap = builder.op_less_than(idx, builder.ir_constant_int(max_len), dbg)
            in_range = builder.op_logical_and(in_low, builder.op_logical_and(in_high_stop, in_high_cap, dbg), dbg)

            read_idx = cast(IntegerValue, builder.op_select(builder.op_bool_cast(in_range, dbg), idx, builder.ir_constant_int(0), dbg))
            if use_mux:
                read_num = self._select_by_index_mux(builder, read_idx, src_values, max_len, fallback_num, dbg)
            else:
                read_int = builder.ir_read_memory(segment_id, read_idx)
                read_num = cast(NumberValue, self._materialize_dtype(builder, read_int, the_self))

            for j in range(max_len):
                ptr_is_j = builder.op_equal(write_ptr, builder.ir_constant_int(j), dbg)
                should_write_j = builder.op_logical_and(in_range, builder.op_bool_cast(ptr_is_j, dbg), dbg)
                out_values[j] = cast(
                    NumberValue,
                    builder.op_select(builder.op_bool_cast(should_write_j, dbg), read_num, out_values[j], dbg),
                )

            next_ptr = cast(IntegerValue, builder.op_add(write_ptr, builder.ir_constant_int(1), dbg))
            write_ptr = builder.ir_select_i(builder.op_bool_cast(in_range, dbg), next_ptr, write_ptr, dbg)

        return DynamicNDArrayValue.from_max_bounds_and_vector(
            max_len,
            the_self.max_rank(),
            the_self.dtype(),
            out_values,
            logical_shape=(max_len,),
            logical_offset=0,
            logical_strides=(1,),
            runtime_logical_length=write_ptr,
        )

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        slicing_params = kwargs["slicing_params"]
        if not isinstance(the_self, DynamicNDArrayValue):
            raise TypeInferenceError(dbg, "Param `self` must be DynamicNDArray")
        if not isinstance(slicing_params, ListValue):
            raise TypeInferenceError(dbg, "Param `slicing_params` must be list")

        params = slicing_params.values()
        if len(params) != 1:
            raise TypeInferenceError(dbg, "DynamicNDArray __getitem__ currently supports 1D indexing/slicing")

        sp = params[0]
        if isinstance(sp, IntegerValue):
            if sp.val(builder) is not None:
                idx = sp.val(builder)
                assert idx is not None
                if idx < 0:
                    idx += the_self.max_length()
                if idx < 0 or idx >= the_self.max_length():
                    raise TypeInferenceError(dbg, "Slicing index out of range")
                return the_self.flattened_values()[idx]
            return self._dynamic_scalar_index(builder, the_self, sp, dbg)

        if isinstance(sp, TupleValue):
            return self._dynamic_slice(builder, the_self, sp, dbg)

        raise TypeInferenceError(dbg, "Unsupported slicing parameter for DynamicNDArray")
