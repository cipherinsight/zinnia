from typing import List, Optional, Tuple, cast

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, IntegerValue, ListValue, NoneValue, NumberValue, TupleValue, Value
from zinnia.compile.type_sys import NumberDTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.op_def.dynamic_ndarray.view_utils import (
    default_like,
    logical_decode_coords,
    logical_encode_coords,
    logical_num_elements,
    logical_row_major_strides,
)


class DynamicNDArray_ConcatenateOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "DynamicNDArray.concatenate"

    @classmethod
    def get_name(cls) -> str:
        return "concatenate"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("arrays"),
            AbstractOp._ParamEntry("axis", default=True),
        ]

    @staticmethod
    def _normalize_axis(
        builder: IRBuilderInterface,
        axis: IntegerValue | NoneValue,
        rank: int,
        dbg: Optional[DebugInfo],
    ) -> Tuple[IntegerValue, int | None]:
        if isinstance(axis, NoneValue):
            return builder.ir_constant_int(0), 0
        if not isinstance(axis, IntegerValue):
            raise TypeInferenceError(dbg, f"Expected `axis` to be an integer, but got {axis.type()}")

        axis_val = axis.val(builder)
        if axis_val is not None:
            normalized = axis_val if axis_val >= 0 else rank + axis_val
            if normalized < 0 or normalized >= rank:
                raise TypeInferenceError(dbg, f"`axis` ({axis_val}) is out of bounds for array of dimension {rank}")
            return builder.ir_constant_int(normalized), normalized

        is_negative = builder.op_less_than(axis, builder.ir_constant_int(0))
        normalized_axis = builder.ir_select_i(
            builder.op_bool_cast(is_negative),
            cast(IntegerValue, builder.op_add(axis, builder.ir_constant_int(rank))),
            axis,
        )
        ge_zero = builder.op_greater_than_or_equal(normalized_axis, builder.ir_constant_int(0))
        lt_rank = builder.op_less_than(normalized_axis, builder.ir_constant_int(rank))
        axis_in_range = builder.op_logical_and(ge_zero, lt_rank)
        builder.op_assert(axis_in_range, builder.op_constant_none(), dbg)
        return normalized_axis, None

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        arrays = kwargs["arrays"]
        axis = kwargs.get("axis", builder.op_constant_none())

        if not isinstance(arrays, (TupleValue, ListValue)):
            raise TypeInferenceError(dbg, f"Expected `arrays` to be a list or tuple, but got {arrays.type()}")
        sources = arrays.values()
        if len(sources) == 0:
            raise TypeInferenceError(dbg, f"At least one array is required for {self.get_name()}")
        if not all(isinstance(src, DynamicNDArrayValue) for src in sources):
            raise TypeInferenceError(dbg, "All input arrays must be DynamicNDArray")

        dyn_sources = cast(List[DynamicNDArrayValue], list(sources))
        rank = len(dyn_sources[0].logical_shape())
        normalized_axis, static_axis = self._normalize_axis(builder, cast(IntegerValue | NoneValue, axis), rank, dbg)

        expected_dtype = dyn_sources[0].dtype()
        for src in dyn_sources:
            if src.dtype() != expected_dtype:
                raise TypeInferenceError(dbg, "Cannot perform concatenate: all input arrays must have the same dtype")

        # Axes that satisfy concatenate shape-compatibility based on current logical metadata.
        valid_axes = []
        for ax in range(rank):
            ok = True
            for i, src in enumerate(dyn_sources):
                if i == 0:
                    continue
                cur_shape = src.logical_shape()
                prev_shape = dyn_sources[i - 1].logical_shape()
                if not all([a == b or j == ax for j, (a, b) in enumerate(zip(cur_shape, prev_shape))]):
                    ok = False
                    break
            if ok:
                valid_axes.append(ax)

        if len(valid_axes) == 0:
            raise TypeInferenceError(dbg, "Cannot perform concatenate: no valid axis satisfies shape-compatibility across sources")

        if static_axis is not None and static_axis not in valid_axes:
            raise TypeInferenceError(dbg, "Cannot perform concatenate: selected axis is not shape-compatible")

        if static_axis is None:
            axis_ok = builder.ir_constant_bool(False)
            for ax in valid_axes:
                is_ax = builder.op_equal(normalized_axis, builder.ir_constant_int(ax))
                axis_ok = builder.op_logical_or(axis_ok, builder.op_bool_cast(is_ax))
            builder.op_assert(axis_ok, builder.op_constant_none(), dbg)

        out_shape_by_axis: dict[int, Tuple[int, ...]] = {}
        out_numel_by_axis: dict[int, int] = {}
        out_row_strides_by_axis: dict[int, Tuple[int, ...]] = {}
        axis_prefix_by_axis: dict[int, List[int]] = {}
        out_numel = 0
        for ax in valid_axes:
            out_shape = list(dyn_sources[0].logical_shape())
            out_shape[ax] = sum(src.logical_shape()[ax] for src in dyn_sources)
            out_shape_t = tuple(out_shape)
            out_shape_by_axis[ax] = out_shape_t
            out_numel_by_axis[ax] = logical_num_elements(out_shape_t)
            out_row_strides_by_axis[ax] = logical_row_major_strides(out_shape_t)

            running = 0
            prefixes = []
            for src in dyn_sources:
                prefixes.append(running)
                running += src.logical_shape()[ax]
            axis_prefix_by_axis[ax] = prefixes
            out_numel = max(out_numel, out_numel_by_axis[ax])

        if static_axis is not None:
            out_shape_t = out_shape_by_axis[static_axis]
            out_numel = out_numel_by_axis[static_axis]
        else:
            out_shape_t = (out_numel,)

        segment_id = len(getattr(builder, "stmts"))
        builder.ir_allocate_memory(segment_id=segment_id, size=out_numel, init_value=0)

        sample_values = dyn_sources[0].flattened_values()
        sample_default = (
            default_like(builder, cast(NumberValue, sample_values[0]))
            if len(sample_values) > 0
            else builder.ir_constant_int(0)
        )

        out_values: List[NumberValue] = []
        for i in range(out_numel):
            per_axis_values: dict[int, NumberValue] = {}
            for ax in valid_axes:
                if i >= out_numel_by_axis[ax]:
                    per_axis_values[ax] = sample_default
                    continue

                out_shape_ax = out_shape_by_axis[ax]
                coords = logical_decode_coords(i, out_shape_ax, out_row_strides_by_axis[ax])
                out_axis = coords[ax]
                prefixes = axis_prefix_by_axis[ax]

                picked = 0
                for idx, src in enumerate(dyn_sources):
                    start = prefixes[idx]
                    end = start + src.logical_shape()[ax]
                    if start <= out_axis < end:
                        picked = idx
                        break

                src = dyn_sources[picked]
                src_coords = list(coords)
                src_coords[ax] = out_axis - prefixes[picked]
                src_linear = src.logical_offset() + logical_encode_coords(src_coords, src.logical_strides())

                src_values = src.flattened_values()
                if len(src_values) == 0:
                    per_axis_values[ax] = builder.ir_constant_int(0)
                elif 0 <= src_linear < len(src_values):
                    per_axis_values[ax] = src_values[src_linear]
                else:
                    per_axis_values[ax] = default_like(builder, cast(NumberValue, src_values[0]))

            src_value = per_axis_values[static_axis] if static_axis is not None else per_axis_values[valid_axes[0]]
            if static_axis is None:
                for ax in valid_axes[1:]:
                    cond = builder.op_equal(normalized_axis, builder.ir_constant_int(ax))
                    src_value = cast(NumberValue, builder.op_select(builder.op_bool_cast(cond), per_axis_values[ax], src_value))

            builder.ir_write_memory(segment_id, builder.ir_constant_int(i), src_value)
            out_values.append(cast(NumberValue, src_value))

        return DynamicNDArrayValue.from_max_bounds_and_vector(
            out_numel,
            max(src.max_rank() for src in dyn_sources),
            cast(NumberDTDescriptor, expected_dtype),
            out_values,
            logical_shape=out_shape_t,
        )
