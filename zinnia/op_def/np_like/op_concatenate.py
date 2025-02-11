from typing import Optional, List, Dict

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, IntegerValue, NDArrayValue, TupleValue, ListValue, NoneValue


class NP_ConcatenateOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.concatenate"

    @classmethod
    def get_name(cls) -> str:
        return "concatenate"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("arrays"),
            AbstractOp._ParamEntry("axis", default=True),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        axis = kwargs.get("axis", builder.op_constant_none())
        arrays = kwargs["arrays"]
        axis_value = 0
        if not isinstance(arrays, TupleValue) and not isinstance(arrays, ListValue):
            raise TypeInferenceError(dbg, f"Expected arrays to be a list or tuple, but got {arrays.type()}")
        if len(arrays.types()) == 0:
            raise TypeInferenceError(dbg, f"At least one array is required for {self.get_name()}")
        expected_float = False
        sources: List[NDArrayValue] = []
        for ary in arrays.values():
            if isinstance(ary, ListValue):
                sources.append(builder.op_np_asarray(ary, dbg))
            elif not isinstance(ary, NDArrayValue):
                raise TypeInferenceError(dbg, f"Expected all arguments to be NDArray, but got {ary.type()}")
            else:
                sources.append(builder.op_np_asarray(ary, dbg))
        for ary in sources:
            if ary.dtype() == FloatType:
                expected_float = True
        sources = [
            builder.op_ndarray_astype(ary, builder.op_constant_class(FloatType)) if expected_float and ary.dtype() != FloatType else ary
            for ary in sources
        ]
        if not isinstance(axis, NoneValue):
            if isinstance(axis, IntegerValue):
                if axis.val() is None:
                    raise StaticInferenceError(dbg, f"`axis` value is not statically inferable")
                axis_value = axis.val() if axis.val() >= 0 else len(sources[0].shape()) + axis.val()
            else:
                raise TypeInferenceError(dbg, f"Expected `axis` to be an integer, but got {axis.type()}")
        for i, src in enumerate(sources):
            if i == 0:
                continue
            lhs_shape = src.shape()
            rhs_shape = sources[i - 1].shape()
            if not len(lhs_shape) == len(rhs_shape):
                raise TypeInferenceError(dbg, "Cannot perform concatenate: elements shape number of dimensions mismatch")
            if len(lhs_shape) <= axis_value or axis_value < 0:
                raise TypeInferenceError(dbg, f"Cannot perform concatenate: `axis` ({axis.val()}) out of bounds for array with {len(lhs_shape)} dimensions")
            if not all([a == b or j == axis_value for j, (a, b) in enumerate(zip(lhs_shape, rhs_shape))]):
                raise TypeInferenceError(dbg, "Cannot perform concatenate: all the input array dimensions except for the concatenation axis must match exactly")
        return NDArrayValue.concatenate(FloatType if expected_float else IntegerType, axis_value, sources)
