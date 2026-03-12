from itertools import permutations
from typing import List, Optional, Tuple, cast

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, IntegerValue, ListValue, NoneValue, NumberValue, TupleValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class DynamicNDArray_TransposeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "DynamicNDArray.transpose"

    @classmethod
    def get_name(cls) -> str:
        return "transpose"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("axes", True),
        ]

    @staticmethod
    def _row_major_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return DynamicNDArrayValue._default_strides(shape)

    @staticmethod
    def _decode_coords(linear: int, shape: Tuple[int, ...], strides: Tuple[int, ...]) -> List[int]:
        return [((linear // stride) % dim) for dim, stride in zip(shape, strides)]

    @staticmethod
    def _encode_coords(coords: List[int], strides: Tuple[int, ...]) -> int:
        out = 0
        for c, s in zip(coords, strides):
            out += c * s
        return out

    @staticmethod
    def _normalize_perm(rank: int, axes: Tuple[int, ...]) -> Tuple[int, ...]:
        out = tuple(ax + rank if ax < 0 else ax for ax in axes)
        if len(out) != rank:
            raise ValueError("Invalid permutation length")
        if sorted(out) != list(range(rank)):
            raise ValueError("Invalid permutation")
        return out

    def _build_transposed_values(
        self,
        builder: IRBuilderInterface,
        arr: DynamicNDArrayValue,
        perm: Tuple[int, ...],
    ) -> List[NumberValue]:
        shape = arr.logical_shape()
        strides = arr.logical_strides()
        offset = arr.logical_offset()
        in_values = arr.flattened_values()
        max_len = arr.max_length()

        out_values: List[NumberValue] = []
        for i in range(max_len):
            coords_out = self._decode_coords(i, tuple(shape[p] for p in perm), self._row_major_strides(tuple(shape[p] for p in perm)))
            coords_in = [0 for _ in range(len(shape))]
            for out_axis, in_axis in enumerate(perm):
                coords_in[in_axis] = coords_out[out_axis]
            src_idx = offset + self._encode_coords(coords_in, strides)
            out_values.append(in_values[src_idx] if 0 <= src_idx < len(in_values) else builder.ir_constant_int(0))
        return out_values

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        axes = kwargs.get("axes", builder.op_constant_none())
        if not isinstance(the_self, DynamicNDArrayValue):
            raise TypeInferenceError(dbg, "Param `self` must be DynamicNDArray")

        rank = len(the_self.logical_shape())
        if isinstance(axes, NoneValue):
            perm = tuple(range(rank - 1, -1, -1))
            new_shape = tuple(the_self.logical_shape()[p] for p in perm)
            new_strides = tuple(the_self.logical_strides()[p] for p in perm)
            return DynamicNDArrayValue.from_max_bounds_and_vector(
                the_self.max_length(),
                the_self.max_rank(),
                the_self.dtype(),
                the_self.flattened_values(),
                logical_shape=new_shape,
                logical_offset=the_self.logical_offset(),
                logical_strides=new_strides,
            )

        if not isinstance(axes, (TupleValue, ListValue)):
            raise TypeInferenceError(dbg, "`axes` should be a tuple/list")

        axes_vals = axes.values()
        has_dynamic_axis = any(isinstance(v, IntegerValue) and v.val(builder) is None for v in axes_vals)

        if not has_dynamic_axis:
            perm = []
            for v in axes_vals:
                if not isinstance(v, IntegerValue):
                    raise TypeInferenceError(dbg, "Each axis in `axes` must be integer")
                ax = v.val(builder)
                if ax is None:
                    raise TypeInferenceError(dbg, "Axis value must be inferrable")
                perm.append(ax)
            try:
                norm_perm = self._normalize_perm(rank, tuple(perm))
            except ValueError as e:
                raise TypeInferenceError(dbg, str(e))
            new_shape = tuple(the_self.logical_shape()[p] for p in norm_perm)
            new_strides = tuple(the_self.logical_strides()[p] for p in norm_perm)
            return DynamicNDArrayValue.from_max_bounds_and_vector(
                the_self.max_length(),
                the_self.max_rank(),
                the_self.dtype(),
                the_self.flattened_values(),
                logical_shape=new_shape,
                logical_offset=the_self.logical_offset(),
                logical_strides=new_strides,
            )

        # Runtime-dynamic axes: select among all static permutations and materialize
        # selected values into flat bounded storage without additional zkRAM IO.
        candidates = list(permutations(range(rank)))
        candidate_values = [self._build_transposed_values(builder, the_self, perm) for perm in candidates]

        selected_values: List[NumberValue] = []
        for i in range(the_self.max_length()):
            cur = candidate_values[0][i]
            for c_idx, perm in enumerate(candidates[1:], start=1):
                is_match = builder.ir_constant_bool(True)
                for ax_i, in_axis in enumerate(perm):
                    arg_axis = cast(IntegerValue, axes_vals[ax_i])
                    is_eq = builder.op_equal(arg_axis, builder.ir_constant_int(in_axis))
                    is_match = builder.op_logical_and(is_match, builder.op_bool_cast(is_eq))
                cur = cast(NumberValue, builder.op_select(builder.op_bool_cast(is_match), candidate_values[c_idx][i], cur))
            selected_values.append(cur)

        return DynamicNDArrayValue.from_max_bounds_and_vector(
            the_self.max_length(),
            the_self.max_rank(),
            the_self.dtype(),
            selected_values,
            logical_shape=(the_self.max_length(),),
            logical_offset=0,
            logical_strides=(1,),
        )
