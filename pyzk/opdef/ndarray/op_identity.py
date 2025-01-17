from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import FloatType, IntegerType
from pyzk.algo.ndarray_helper import NDArrayValueWrapper
from pyzk.debug.dbg_info import DebugInfo
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, IntegerValue, ClassValue, NDArrayValue


class NDArray_IdentityOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::identity"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::identity"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("n")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        n = kwargs["n"]
        dtype = kwargs["dtype"]
        if not isinstance(n, IntegerValue):
            raise TypeInferenceError(dbg, "Param `n` must be of type `Number`")
        if n.val() is None:
            raise StaticInferenceError(dbg, "Cannot statically infer the value of param `n`")
        if n.val() <= 0:
            raise TypeInferenceError(dbg, "Invalid `n` value, n must be greater than 0")
        parsed_dtype = FloatType
        if dtype is not None:
            if isinstance(dtype, ClassValue):
                parsed_dtype = dtype.val()
            else:
                raise TypeInferenceError(dbg, f"Invalid argument dtype, it must be a datatype")
        result_shape = (n.val(), n.val())
        constant_0_i = reducer.ir_constant_int(0)
        constant_1_i = reducer.ir_constant_int(1)
        constant_0_f = reducer.ir_constant_float(0.0)
        constant_1_f = reducer.ir_constant_float(1.0)
        if parsed_dtype == FloatType:
            ndarray = NDArrayValueWrapper.fill(result_shape, lambda: constant_0_f)
            ndarray = ndarray.for_each(lambda pos, val: constant_1_f if pos[0] == pos[1] else constant_0_f)
        elif parsed_dtype == IntegerType:
            ndarray = NDArrayValueWrapper.fill(result_shape, lambda: constant_0_i)
            ndarray = ndarray.for_each(lambda pos, val: constant_1_i if pos[0] == pos[1] else constant_0_i)
        else:
            raise TypeInferenceError(dbg, f"Unsupported NDArray dtype {parsed_dtype}")
        return NDArrayValue(result_shape, parsed_dtype, ndarray)
