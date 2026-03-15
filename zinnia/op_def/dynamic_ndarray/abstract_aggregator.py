from itertools import product
from typing import List, Optional, Tuple, cast

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, IntegerValue, NoneValue, NumberValue, Value
from zinnia.compile.type_sys import NumberDTDescriptor
from zinnia.compile.type_sys.dt_descriptor import DTDescriptor
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class DynamicAbstractAggregator(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("axis", default=True),
        ]

    def aggregator_func(
        self,
        builder: IRBuilderInterface,
        lhs: NumberValue,
        lhs_i: NumberValue,
        rhs: NumberValue,
        rhs_i: NumberValue,
        dt: DTDescriptor,
    ) -> Tuple[NumberValue, NumberValue | None]:
        raise NotImplementedError()

    def depair_func(self, builder: IRBuilderInterface, a: NumberValue, b: NumberValue) -> NumberValue:
        return a

    def get_result_dtype(self, element_dt: DTDescriptor):
        return element_dt

    def is_allowed_ndarray_dtype(self, element_dt: DTDescriptor) -> bool:
        return True

    @staticmethod
    def _iter_indices(shape: Tuple[int, ...]):
        if len(shape) == 0:
            yield ()
            return
        for idx in product(*[range(dim) for dim in shape]):
            yield idx

    @staticmethod
    def _flat_index(idx: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
        mul = 1
        out = 0
        for i in range(len(shape) - 1, -1, -1):
            out += idx[i] * mul
            mul *= shape[i]
        return out

    def _reduce_for_axis(
        self,
        builder: IRBuilderInterface,
        values: List[NumberValue],
        logical_shape: Tuple[int, ...],
        axis: int | None,
        out_dtype: DTDescriptor,
    ) -> Value:
        if axis is None:
            first_ele = values[0]
            acc = first_ele
            acc_i = builder.ir_constant_int(0)
            inited = False
            for idx, value in enumerate(values):
                if not inited:
                    acc = value
                    acc_i = builder.ir_constant_int(idx)
                    inited = True
                    continue
                acc, cand_i = self.aggregator_func(builder, acc, acc_i, value, builder.ir_constant_int(idx), out_dtype)
                if cand_i is not None:
                    acc_i = cand_i
            return self.depair_func(builder, acc, acc_i)

        out_shape = logical_shape[:axis] + logical_shape[axis + 1:]
        reduced_values: List[NumberValue] = []
        for out_idx in self._iter_indices(out_shape):
            acc = values[self._flat_index(out_idx[:axis] + (0,) + out_idx[axis:], logical_shape)]
            acc_i = builder.ir_constant_int(0)
            inited = False
            for axis_i in range(logical_shape[axis]):
                full_idx = out_idx[:axis] + (axis_i,) + out_idx[axis:]
                rhs = values[self._flat_index(full_idx, logical_shape)]
                if not inited:
                    acc = rhs
                    acc_i = builder.ir_constant_int(axis_i)
                    inited = True
                    continue
                acc, cand_i = self.aggregator_func(builder, acc, acc_i, rhs, builder.ir_constant_int(axis_i), out_dtype)
                if cand_i is not None:
                    acc_i = cand_i
            reduced_values.append(self.depair_func(builder, acc, acc_i))

        if len(out_shape) == 0:
            return reduced_values[0]

        return DynamicNDArrayValue.from_max_bounds_and_vector(
            len(reduced_values),
            max(len(out_shape), 1),
            cast(NumberDTDescriptor, out_dtype),
            reduced_values,
            logical_shape=(len(reduced_values),),
        )

    def _reduce_dynamic_axis(
        self,
        builder: IRBuilderInterface,
        values: List[NumberValue],
        logical_shape: Tuple[int, ...],
        axis: IntegerValue,
        out_dtype: DTDescriptor,
        dbg: Optional[DebugInfo] = None,
    ) -> Value:
        rank = len(logical_shape)
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

        candidates: List[Value] = [self._reduce_for_axis(builder, values, logical_shape, ax, out_dtype) for ax in range(rank)]
        if isinstance(candidates[0], NumberValue):
            selected = candidates[0]
            for ax in range(1, rank):
                pick_axis = builder.op_equal(normalized_axis, builder.ir_constant_int(ax))
                selected = self.depair_func(
                    builder,
                    cast(NumberValue, builder.op_select(builder.op_bool_cast(pick_axis), candidates[ax], selected)),
                    builder.ir_constant_int(0),
                )
            return selected

        max_len = 0
        max_rank = 1
        selected_values: List[NumberValue] = []
        for candidate in candidates:
            assert isinstance(candidate, DynamicNDArrayValue)
            max_len = max(max_len, candidate.max_length())
            max_rank = max(max_rank, candidate.max_rank())

        for i in range(max_len):
            first_candidate = candidates[0]
            assert isinstance(first_candidate, DynamicNDArrayValue)
            first_vals = first_candidate.flattened_values()
            if len(first_vals) == 0:
                cur = builder.ir_constant_int(0)
            elif i < len(first_vals):
                cur = first_vals[i]
            else:
                cur = builder.ir_constant_bool(False) if first_vals[0].type() == builder.ir_constant_bool(False).type() else builder.ir_constant_int(0)
            for ax in range(1, rank):
                candidate = candidates[ax]
                assert isinstance(candidate, DynamicNDArrayValue)
                cand_vals = candidate.flattened_values()
                if len(cand_vals) == 0:
                    cand_val = builder.ir_constant_int(0)
                elif i < len(cand_vals):
                    cand_val = cand_vals[i]
                else:
                    cand_val = builder.ir_constant_bool(False) if cand_vals[0].type() == builder.ir_constant_bool(False).type() else builder.ir_constant_int(0)
                pick_axis = builder.op_equal(normalized_axis, builder.ir_constant_int(ax))
                cur = cast(NumberValue, builder.op_select(builder.op_bool_cast(pick_axis), cand_val, cur))
            selected_values.append(cast(NumberValue, cur))

        return DynamicNDArrayValue.from_max_bounds_and_vector(
            max_len,
            max_rank,
            cast(NumberDTDescriptor, out_dtype),
            selected_values,
            logical_shape=(max_len,),
        )

    def _emit_oblivious_memory_trace(
        self,
        builder: IRBuilderInterface,
        arr: DynamicNDArrayValue,
        axis: IntegerValue | None,
        dbg: Optional[DebugInfo],
    ) -> None:
        shape = arr.logical_shape()
        strides = arr.logical_strides()
        offset = arr.logical_offset()
        rank = len(shape)
        max_len = arr.max_length()

        if rank == 0:
            return

        if axis is None:
            normalized_axis = builder.ir_constant_int(0)
        else:
            is_negative = builder.op_less_than(axis, builder.ir_constant_int(0))
            normalized_axis = builder.ir_select_i(
                builder.op_bool_cast(is_negative),
                cast(IntegerValue, builder.op_add(axis, builder.ir_constant_int(rank))),
                axis,
            )

        trace_segment = len(getattr(builder, "stmts"))
        builder.ir_allocate_memory(trace_segment, max_len, 0)

        for i in range(max_len):
            linear = i + offset
            coords = []
            for dim, stride in zip(shape, strides):
                coords.append((linear // stride) % dim)

            if axis is None:
                target_addr = builder.ir_constant_int(i)
            else:
                target_candidates = []
                for ax in range(rank):
                    out_shape = shape[:ax] + shape[ax + 1:]
                    out_strides = DynamicNDArrayValue._default_strides(out_shape)
                    out_coords = coords[:ax] + coords[ax + 1:]
                    target = 0
                    for c, s in zip(out_coords, out_strides):
                        target += c * s
                    target_candidates.append(target)
                target_addr = builder.ir_constant_int(target_candidates[0])
                for ax in range(1, rank):
                    cond = builder.op_equal(normalized_axis, builder.ir_constant_int(ax))
                    target_addr = builder.ir_select_i(
                        builder.op_bool_cast(cond),
                        builder.ir_constant_int(target_candidates[ax]),
                        target_addr,
                    )

            old_val = builder.ir_read_memory(trace_segment, target_addr)
            builder.ir_write_memory(trace_segment, target_addr, old_val)

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        the_axis = kwargs.get("axis", builder.op_constant_none())
        if not isinstance(the_self, DynamicNDArrayValue):
            raise TypeInferenceError(dbg, "Param `self` must be a DynamicNDArray")
        if not self.is_allowed_ndarray_dtype(the_self.dtype()):
            raise TypeInferenceError(dbg, f"The dtype ({the_self.dtype()}) of param `self: DynamicNDArray` is not allowed here")

        values = the_self.flattened_values()
        out_dtype = self.get_result_dtype(the_self.dtype())

        if isinstance(the_axis, NoneValue):
            self._emit_oblivious_memory_trace(builder, the_self, None, dbg)
            return self._reduce_for_axis(builder, values, the_self.logical_shape(), None, out_dtype)

        if not isinstance(the_axis, IntegerValue):
            raise TypeInferenceError(dbg, "Param `axis` must be of type `Integer`")

        self._emit_oblivious_memory_trace(builder, the_self, the_axis, dbg)
        if the_axis.val(builder) is None:
            return self._reduce_dynamic_axis(builder, values, the_self.logical_shape(), the_axis, out_dtype, dbg)

        axis_val = the_axis.val(builder)
        assert axis_val is not None
        if axis_val < 0:
            axis_val += len(the_self.logical_shape())
        if axis_val < 0 or axis_val >= len(the_self.logical_shape()):
            raise TypeInferenceError(dbg, f"axis `{axis_val}` is out of bounds for array of dimension {len(the_self.logical_shape())}")
        return self._reduce_for_axis(builder, values, the_self.logical_shape(), axis_val, out_dtype)
