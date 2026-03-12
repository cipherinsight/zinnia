from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, IntegerValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.op_def.dynamic_ndarray.op_transpose import DynamicNDArray_TransposeOp


class DynamicNDArray_MoveAxisOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "DynamicNDArray.moveaxis"

    @classmethod
    def get_name(cls) -> str:
        return "moveaxis"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("source"),
            AbstractOp._ParamEntry("destination"),
        ]

    @staticmethod
    def _perm_from_moveaxis(rank: int, src: int, dst: int) -> tuple[int, ...]:
        src_axis = src if src >= 0 else src + rank
        dst_axis = dst if dst >= 0 else dst + rank
        if src_axis < 0 or src_axis >= rank or dst_axis < 0 or dst_axis >= rank:
            raise ValueError("Invalid axis for moveaxis")
        axes = list(range(rank))
        moved = axes.pop(src_axis)
        axes.insert(dst_axis, moved)
        return tuple(axes)

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        source = kwargs["source"]
        destination = kwargs["destination"]
        if not isinstance(the_self, DynamicNDArrayValue):
            raise TypeInferenceError(dbg, "Param `self` must be DynamicNDArray")
        if not isinstance(source, IntegerValue) or not isinstance(destination, IntegerValue):
            raise TypeInferenceError(dbg, "`source` and `destination` must be integers")

        rank = len(the_self.logical_shape())
        src_val = source.val(builder)
        dst_val = destination.val(builder)

        transpose_op = DynamicNDArray_TransposeOp()
        if src_val is not None and dst_val is not None:
            try:
                perm = self._perm_from_moveaxis(rank, src_val, dst_val)
            except ValueError as e:
                raise TypeInferenceError(dbg, str(e))
            axes = builder.op_parenthesis([builder.ir_constant_int(x) for x in perm])
            kwargs2 = transpose_op.argparse(dbg, [the_self], {"axes": axes})
            return transpose_op.build(builder, OpArgsContainer(kwargs2), dbg)

        # Runtime-dynamic source/destination: route through dynamic transpose with
        # runtime-selected permutation candidates.
        candidate_perms = []
        for src in range(rank):
            for dst in range(rank):
                candidate_perms.append((src, dst, self._perm_from_moveaxis(rank, src, dst)))

            candidate_values = [transpose_op._build_transposed_values(builder, the_self, perm) for _, _, perm in candidate_perms]

            selected_values = []
            for i in range(the_self.max_length()):
                cur = candidate_values[0][i]
                for c_idx, (src, dst, _) in enumerate(candidate_perms[1:], start=1):
                    src_eq = builder.op_equal(source, builder.ir_constant_int(src))
                    dst_eq = builder.op_equal(destination, builder.ir_constant_int(dst))
                    cond = builder.op_logical_and(builder.op_bool_cast(src_eq), builder.op_bool_cast(dst_eq))
                    cur = builder.op_select(builder.op_bool_cast(cond), candidate_values[c_idx][i], cur)
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
