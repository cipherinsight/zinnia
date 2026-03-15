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


class DynamicNDArray_StackOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "DynamicNDArray.stack"

    @classmethod
    def get_name(cls) -> str:
        return "stack"

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
            normalized = axis_val if axis_val >= 0 else rank + axis_val + 1
            if normalized < 0 or normalized > rank:
                raise TypeInferenceError(dbg, f"`axis` ({axis_val}) is out of bounds for array of dimension {rank}")
            return builder.ir_constant_int(normalized), normalized

        is_negative = builder.op_less_than(axis, builder.ir_constant_int(0))
        normalized_axis = builder.ir_select_i(
            builder.op_bool_cast(is_negative),
            cast(IntegerValue, builder.op_add(axis, builder.ir_constant_int(rank + 1))),
            axis,
        )
        ge_zero = builder.op_greater_than_or_equal(normalized_axis, builder.ir_constant_int(0))
        le_rank = builder.op_less_than_or_equal(normalized_axis, builder.ir_constant_int(rank))
        axis_in_range = builder.op_logical_and(ge_zero, le_rank)
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
        source_shapes = [src.logical_shape() for src in dyn_sources]
        rank = max(len(shape) for shape in source_shapes)
        normalized_axis, static_axis = self._normalize_axis(builder, cast(IntegerValue | NoneValue, axis), rank, dbg)

        expected_dtype = dyn_sources[0].dtype()
        for src in dyn_sources:
            if src.dtype() != expected_dtype:
                raise TypeInferenceError(dbg, "Cannot perform stack: all input arrays must have the same dtype")

        # Compute broadcast-compatible base shape across all inputs.
        aligned_shapes: List[Tuple[int, ...]] = []
        for shape in source_shapes:
            pad = rank - len(shape)
            aligned_shapes.append(tuple([1] * pad + list(shape)))

        base_shape: List[int] = [1 for _ in range(rank)]
        for d in range(rank):
            dim = 1
            for ashape in aligned_shapes:
                v = ashape[d]
                if v != 1:
                    if dim == 1:
                        dim = v
                    elif dim != v:
                        raise TypeInferenceError(dbg, "Cannot perform stack: input shapes are not broadcast-compatible")
            base_shape[d] = dim
        base_shape_t = tuple(base_shape)

        axis_candidates = list(range(rank + 1))
        out_shape_by_axis: dict[int, Tuple[int, ...]] = {}
        out_numel_by_axis: dict[int, int] = {}
        out_row_strides_by_axis: dict[int, Tuple[int, ...]] = {}
        out_numel = 0
        for ax in axis_candidates:
            out_shape = base_shape_t[:ax] + (len(dyn_sources),) + base_shape_t[ax:]
            out_shape_by_axis[ax] = out_shape
            out_numel_by_axis[ax] = logical_num_elements(out_shape)
            out_row_strides_by_axis[ax] = logical_row_major_strides(out_shape)
            out_numel = max(out_numel, out_numel_by_axis[ax])

        if static_axis is not None:
            out_shape = out_shape_by_axis[static_axis]
            out_numel = out_numel_by_axis[static_axis]
        else:
            out_shape = (out_numel,)

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
            for ax in axis_candidates:
                if i >= out_numel_by_axis[ax]:
                    per_axis_values[ax] = sample_default
                    continue

                out_shape_ax = out_shape_by_axis[ax]
                out_coords = logical_decode_coords(i, out_shape_ax, out_row_strides_by_axis[ax])
                src_idx = out_coords[ax]
                src = dyn_sources[src_idx]

                base_coords = list(out_coords[:ax] + out_coords[ax + 1:])
                src_shape = src.logical_shape()
                src_rank = len(src_shape)
                src_pad = rank - src_rank
                src_aligned_shape = tuple([1] * src_pad + list(src_shape))

                src_coords_aligned: List[int] = []
                for d in range(rank):
                    src_coords_aligned.append(0 if src_aligned_shape[d] == 1 else base_coords[d])
                src_coords = src_coords_aligned[src_pad:]

                src_linear = src.logical_offset() + logical_encode_coords(src_coords, src.logical_strides())
                src_values = src.flattened_values()
                if len(src_values) == 0:
                    per_axis_values[ax] = builder.ir_constant_int(0)
                elif 0 <= src_linear < len(src_values):
                    per_axis_values[ax] = src_values[src_linear]
                else:
                    per_axis_values[ax] = default_like(builder, cast(NumberValue, src_values[0]))

            src_value = per_axis_values[static_axis] if static_axis is not None else per_axis_values[axis_candidates[0]]
            if static_axis is None:
                for ax in axis_candidates[1:]:
                    cond = builder.op_equal(normalized_axis, builder.ir_constant_int(ax))
                    src_value = cast(NumberValue, builder.op_select(builder.op_bool_cast(cond), per_axis_values[ax], src_value))

            builder.ir_write_memory(segment_id, builder.ir_constant_int(i), src_value)
            out_values.append(cast(NumberValue, src_value))

        return DynamicNDArrayValue.from_max_bounds_and_vector(
            out_numel,
            max(max(src.max_rank() for src in dyn_sources) + 1, len(out_shape)),
            cast(NumberDTDescriptor, expected_dtype),
            out_values,
            logical_shape=out_shape,
        )
