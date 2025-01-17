from typing import Any, Optional, List, Dict

from pyzk.algo.ndarray_helper import NDArrayValueWrapper
from pyzk.debug.dbg_info import DebugInfo
from pyzk.debug.exception import OperatorCallError, TypeInferenceError, StaticInferenceError
from pyzk.internal.dt_descriptor import IntegerType, FloatType
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, NDArrayValue


class StackOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "stack"

    @classmethod
    def get_name(cls) -> str:
        return "stack"

    def argparse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in kwargs.items():
            if key != "axis":
                raise OperatorCallError(dbg_i, f"Unexpected keyword argument {key}")
        axis = kwargs.get("axis", None)
        if len(args) == 0:
            raise OperatorCallError(dbg_i, f"Operator `{self.get_name()}` requires at least one argument")
        return_dict = {f"_n_{i}": arg for i, arg in enumerate(args)}
        return_dict["axis"] = axis
        return return_dict

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        axis = kwargs.get("axis", None)
        args = [v for k, v in kwargs.items() if k.startswith("_n_")]
        if len(args) == 0:
            raise TypeInferenceError(dbg, f"At least one argument is required for {self.get_name()}")
        axis_value = 0
        if axis is not None:
            if axis.type() != IntegerType:
                raise TypeInferenceError(dbg, f"Expected axis to be an integer, but got {axis.type()}")
            axis_value = axis.val()
            if axis_value is None:
                raise StaticInferenceError(dbg, f"Axis value is not statically inferable")
        for arg in args:
            if not isinstance(arg, NDArrayValue):
                raise TypeInferenceError(dbg, f"Expected all arguments to be NDArray, but got {arg.type()}")
        check_stack = NDArrayValueWrapper.check_stack([arg.get() for arg in args], axis_value)
        if check_stack is not None:
            raise TypeInferenceError(dbg, check_stack)
        expected_dtype = IntegerType
        for arg in args:
            expected_dtype = FloatType if arg.dtype() == FloatType else expected_dtype
        if expected_dtype == FloatType:
            args = [(reducer.op_float_cast(arg) if arg.dtype() == IntegerType else arg) for arg in args]
        result = NDArrayValueWrapper.stack([arg.get() for arg in args], axis_value)
        return NDArrayValue(result.shape, expected_dtype, result)
