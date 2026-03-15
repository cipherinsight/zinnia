import copy
from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import IntegerValue, NDArrayValue, DynamicNDArrayValue, Value
from zinnia.compile.type_sys.ndarray_bounds import infer_ndarray_compile_bounds_from_static_shape
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import StaticInferenceError, TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.op_def.dynamic_ndarray.op_moveaxis import DynamicNDArray_MoveAxisOp


class NDArray_MoveAxisOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.moveaxis"

    @classmethod
    def get_name(cls) -> str:
        return "moveaxis"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("source"),
            AbstractOp._ParamEntry("destination"),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        source = kwargs["source"]
        destination = kwargs["destination"]
        assert isinstance(the_self, NDArrayValue)

        if isinstance(the_self, DynamicNDArrayValue):
            op = DynamicNDArray_MoveAxisOp()
            kwargs2 = op.argparse(dbg, [the_self, source, destination], {})
            return op.build(builder, OpArgsContainer(kwargs2), dbg)

        bounds = infer_ndarray_compile_bounds_from_static_shape(the_self.shape(), dbg, self.get_name())
        if bounds.max_rank == 0:
            raise TypeInferenceError(dbg, "`moveaxis` is not defined for scalar NDArray")

        if not isinstance(source, IntegerValue):
            raise TypeInferenceError(dbg, f"`source` should be an integer, but got {source.type()}")
        if not isinstance(destination, IntegerValue):
            raise TypeInferenceError(dbg, f"`destination` should be an integer, but got {destination.type()}")

        src_val = source.val(builder)
        dst_val = destination.val(builder)
        if src_val is None or dst_val is None:
            op = DynamicNDArray_MoveAxisOp()
            dyn_self = the_self.to_dynamic_ndarray()
            kwargs2 = op.argparse(dbg, [dyn_self, source, destination], {})
            return op.build(builder, OpArgsContainer(kwargs2), dbg)

        ndim = len(the_self.shape())
        src_axis = src_val if src_val >= 0 else src_val + ndim
        dst_axis = dst_val if dst_val >= 0 else dst_val + ndim
        if src_axis < 0 or src_axis >= ndim:
            raise TypeInferenceError(dbg, f"Invalid source axis `{src_val}` for array of dimension {ndim}")
        if dst_axis < 0 or dst_axis >= ndim:
            raise TypeInferenceError(dbg, f"Invalid destination axis `{dst_val}` for array of dimension {ndim}")

        axes = list(range(ndim))
        moved_axis = axes.pop(src_axis)
        axes.insert(dst_axis, moved_axis)
        return copy.deepcopy(the_self.transpose(tuple(axes)))