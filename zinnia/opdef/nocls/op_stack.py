from typing import Any, Optional, List, Dict

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import OperatorCallError, TypeInferenceError, StaticInferenceError
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, NDArrayValue


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

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        axis = kwargs.get("axis", None)
        args: List[NDArrayValue] = [v for k, v in kwargs.items() if k.startswith("_n_")]
        if len(args) == 0:
            raise TypeInferenceError(dbg, f"At least one argument is required for {self.get_name()}")
        for arg in args:
            if not isinstance(arg, NDArrayValue):
                raise TypeInferenceError(dbg, f"Expected all arguments to be NDArray, but got {arg.type()}")
        axis_value = 0
        if axis is not None:
            if axis.type() != IntegerType:
                raise TypeInferenceError(dbg, f"Expected axis to be an integer, but got {axis.type()}")
            axis_value = axis.val()
            if axis_value is None:
                raise StaticInferenceError(dbg, f"Axis value is not statically inferable")
        for i, arg in enumerate(args):
            if not (i == 0 or arg.shape() == args[i - 1].shape()):
                raise TypeInferenceError(dbg, f"Cannot perform stack: all input arrays must have the same shape")
            if axis_value < 0 or axis_value > len(arg.shape()):
                raise TypeInferenceError(dbg, f"Axis {axis_value} is out of bounds for array of dimension {len(arg.shape())}")
        expected_dtype = IntegerType
        for arg in args:
            expected_dtype = FloatType if arg.dtype() == FloatType else expected_dtype
        if expected_dtype == FloatType:
            args = [(builder.op_float_cast(arg) if arg.dtype() == IntegerType else arg) for arg in args]
        return NDArrayValue.stack(expected_dtype, axis_value, args)
