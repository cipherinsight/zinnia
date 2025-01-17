from typing import Optional, List, Dict

from zenopy.algo.ndarray_helper import NDArrayValueWrapper
from zenopy.debug.dbg_info import DebugInfo
from zenopy.debug.exception import TypeInferenceError, StaticInferenceError
from zenopy.internal.dt_descriptor import IntegerType, FloatType
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue, NDArrayValue, TupleValue


class ConcatenateOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "concatenate"

    @classmethod
    def get_name(cls) -> str:
        return "concatenate"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("args"),
            AbstractOp._ParamEntry("axis", default=True),
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        axis = kwargs.get("axis", None)
        args = kwargs["args"]
        axis_value = 0
        if axis is not None:
            if isinstance(axis, IntegerValue):
                axis_value = axis.val()
            else:
                raise TypeInferenceError(dbg, f"Expected axis to be an integer, but got {axis.type()}")
        if axis_value is None:
            raise StaticInferenceError(dbg, f"`axis` value is not statically inferable")
        if not isinstance(args, TupleValue):
            raise TypeInferenceError(dbg, f"Expected args to be a tuple, but got {args.type()}")
        if len(args.types()) == 0:
            raise TypeInferenceError(dbg, f"At least one array is required for {self.get_name()}")
        expected_float = False
        for arg in args.values():
            if not isinstance(arg, NDArrayValue):
                raise TypeInferenceError(dbg, f"Expected all arguments to be NDArray, but got {arg.type()}")
            if arg.dtype() == FloatType:
                expected_float = True
        sources: List[NDArrayValue] = []
        for arg in args.values():
            sources.append(reducer.op_float_cast(arg) if expected_float and arg.dtype() != FloatType else arg)
        check_result = NDArrayValueWrapper.check_concatenate([source.get() for source in sources])
        if check_result is not None:
            raise TypeInferenceError(dbg, check_result)
        result = NDArrayValueWrapper.concatenate([source.get() for source in sources], axis_value)
        return NDArrayValue(result.shape, FloatType if expected_float else IntegerType, result)
