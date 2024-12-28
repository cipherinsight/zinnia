from typing import Any, Optional, List, Dict

from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo
from pyzk.debug.exception import OperatorCallError, TypeInferenceError, StaticInferenceError
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, IntegerInferenceDescriptor
from pyzk.opdef.nocls.abstract_op import AbstractOp


class StackOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "stack"

    @classmethod
    def get_name(cls) -> str:
        return "stack"

    def params_parse(self, dbg_i: Optional[DebugInfo], args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in kwargs.items():
            if key != "axis":
                raise OperatorCallError(dbg_i, f"Unexpected keyword argument {key}")
        axis = kwargs.get("axis", None)
        if len(args) == 0:
            raise OperatorCallError(dbg_i, f"Operator `{self.get_name()}` requires at least one argument")
        return_dict = {f"_n_{i}": arg for i, arg in enumerate(args)}
        return_dict["axis"] = axis
        return return_dict

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        axis = kwargs.get("axis", None)
        args = [v for k, v in kwargs.items() if k.startswith("_n_")]
        axis_value = 0
        if len(args) == 0:
            raise TypeInferenceError(dbg_i, f"At least one argument is required for {self.get_name()}")
        if axis is not None and not isinstance(axis, IntegerInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f"Expected axis to be an integer, but got {axis}")
        if axis is not None:
            axis_value = axis.get()
            if axis_value is None:
                raise StaticInferenceError(dbg_i, f"Axis value is not statically inferable")
        for arg in args:
            if not isinstance(arg, NDArrayInferenceDescriptor):
                raise TypeInferenceError(dbg_i, f"Expected all arguments to be NDArray, but got {arg}")
        check_stack = NDArrayHelper.check_stack([arg.get() for arg in args], axis_value)
        if check_stack is not None:
            raise TypeInferenceError(dbg_i, check_stack)
        return NDArrayDTDescriptor(NDArrayHelper.stack_shape([arg.get() for arg in args], axis_value), args[0].dtype())

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        axis = kwargs.get("axis", None)
        args = [v for k, v in kwargs.items() if k.startswith("_n_")]
        axis_value = 0
        if axis is not None:
            axis_value = axis.get()
        assert axis_value is not None
        result = NDArrayHelper.stack([arg.get() for arg in args], axis_value)
        return NDArrayInferenceDescriptor(result.shape, args[0].dtype(), result)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        axis = kwargs.get("axis", None)
        args = [v for k, v in kwargs.items() if k.startswith("_n_")]
        axis_value = 0
        if axis is not None:
            axis_value = axis.val()
        assert axis_value is not None
        result = NDArrayHelper.stack([arg.ptr() for arg in args], axis_value)
        return NDArrayFlattenDescriptor(result.shape, args[0].dtype(), result)
