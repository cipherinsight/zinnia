from typing import List, Dict, Optional

from zenopy.debug.exception import TypeInferenceError, StaticInferenceError
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.internal.dt_descriptor import FloatType, IntegerType
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, IntegerValue, ClassValue, NDArrayValue


class NDArray_EyeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::eye"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::eye"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("n"),
            AbstractOp._ParamEntry("m"),
            AbstractOp._ParamEntry("dtype", True),
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        n = kwargs["n"]
        m = kwargs["m"]
        dtype = kwargs["dtype"]
        if not isinstance(n, IntegerValue):
            raise TypeInferenceError(dbg, "Param `n` must be of type `Number`")
        if n.val() is None:
            raise StaticInferenceError(dbg, "Cannot statically infer the value of param `n`")
        if n.val() <= 0:
            raise TypeInferenceError(dbg, "Invalid `n` value, n must be greater than 0")
        if not isinstance(m, IntegerValue):
            raise TypeInferenceError(dbg, "Param `m` must be of type `Number`")
        if m.val() is None:
            raise StaticInferenceError(dbg, "Cannot statically infer the value of param `m`")
        if m.val() <= 0:
            raise TypeInferenceError(dbg, "Invalid `m` value, m must be greater than 0")
        parsed_dtype = FloatType
        if dtype is not None:
            if isinstance(dtype, ClassValue):
                parsed_dtype = dtype.val()
            else:
                raise TypeInferenceError(dbg, f"Invalid argument dtype, it must be a datatype")
        result_shape = (n.val(), m.val())
        if parsed_dtype == FloatType:
            ndarray = NDArrayValue.fill(result_shape, FloatType, lambda: reducer.ir_constant_float(0.0))
            ndarray = ndarray.for_each(lambda pos, val: reducer.ir_constant_float(1.0) if pos[0] == pos[1] else reducer.ir_constant_float(0.0))
        elif parsed_dtype == IntegerType:
            ndarray = NDArrayValue.fill(result_shape, IntegerType, lambda: reducer.ir_constant_int(0))
            ndarray = ndarray.for_each(lambda pos, val: reducer.ir_constant_int(1) if pos[0] == pos[1] else reducer.ir_constant_int(0))
        else:
            raise TypeInferenceError(dbg, f"Unsupported NDArray dtype {parsed_dtype}")
        return ndarray
