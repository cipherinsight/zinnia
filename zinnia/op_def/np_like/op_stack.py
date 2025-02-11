from typing import Optional, List, Dict

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, TupleValue, ListValue, NoneValue


class NP_StackOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.stack"

    @classmethod
    def get_name(cls) -> str:
        return "stack"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("arrays"),
            AbstractOp._ParamEntry("axis", True)
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_arrays = kwargs['arrays']
        axis = kwargs.get("axis", builder.op_constant_none())
        if not isinstance(the_arrays, TupleValue) and not isinstance(the_arrays, ListValue):
            raise TypeInferenceError(dbg, f"Expected `arrays` to be a list or tuple, but got {the_arrays.type()}")
        arrays = []
        if len(the_arrays.values()) == 0:
            raise TypeInferenceError(dbg, f"At least one array is required for {self.get_name()}. Got 0.")
        for arg in the_arrays.values():
            if isinstance(arg, NDArrayValue):
                arrays.append(arg)
            elif isinstance(arg, ListValue):
                arrays.append(builder.op_np_asarray(arg))
            else:
                raise TypeInferenceError(dbg, f"Expected all arguments to be NDArray, but got {arg.type()}")
        axis_value = 0
        if not isinstance(axis, NoneValue):
            if axis.type() != IntegerType:
                raise TypeInferenceError(dbg, f"Expected `axis` to be an integer, but got {axis.type()}")
            if axis.val() is None:
                raise StaticInferenceError(dbg, f"`axis` value is not statically inferable")
            axis_value = axis.val() if axis.val() >= 0 else len(arrays[0].shape()) + axis.val() + 1
        for i, arg in enumerate(arrays):
            if not (i == 0 or arg.shape() == arrays[i - 1].shape()):
                raise TypeInferenceError(dbg, f"Cannot perform stack: all input arrays must have the same shape")
            if axis_value < 0 or axis_value > len(arg.shape()):
                raise TypeInferenceError(dbg, f"`axis` ({axis.val()}) is out of bounds for array of dimension {len(arg.shape())}")
        expected_dtype = IntegerType
        for arg in arrays:
            expected_dtype = FloatType if arg.dtype() == FloatType else expected_dtype
        if expected_dtype == FloatType:
            arrays = [(builder.op_ndarray_astype(arg, builder.op_constant_class(FloatType)) if arg.dtype() == IntegerType else arg) for arg in arrays]
        return NDArrayValue.stack(expected_dtype, axis_value, arrays)
