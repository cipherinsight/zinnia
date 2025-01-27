from typing import List, Dict, Optional

from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.opdef.nocls.abstract_op import AbstractOp
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.abstract_ir_builder import AbsIRBuilderInterface
from zinnia.compile.builder.value import Value, TupleValue, IntegerValue, ClassValue, NDArrayValue


class NDArray_OnesOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::ones"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::ones"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("shape"),
            AbstractOp._ParamEntry("dtype", True)
        ]

    def build(self, builder: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        shape = kwargs["shape"]
        dtype = kwargs["dtype"]
        if not isinstance(shape, TupleValue):
            raise TypeInferenceError(dbg, "Param `shape` must be of type `Tuple`")
        for ele_t, ele_v in zip(shape.types(), shape.values()):
            if ele_t != IntegerType:
                raise TypeInferenceError(dbg, "Every element in `shape` Tuple must be of type `Integer`")
            assert isinstance(ele_v, IntegerValue)
            if ele_v.val() is None:
                raise StaticInferenceError(dbg, "Every number element in `shape` must be statically inferrable")
            if ele_v.val() <= 0:
                raise TypeInferenceError(dbg, "Every number element in `shape` must be greater than 0")
        parsed_dtype = FloatType
        if dtype is not None:
            if isinstance(dtype, ClassValue):
                parsed_dtype = dtype.val()
            else:
                raise TypeInferenceError(dbg, f"Invalid type for argument `dtype`: {dtype.type()}, it must be a datatype")
        parsed_shape = tuple(v.val() for v in shape.values())
        if parsed_dtype == FloatType:
            return NDArrayValue.fill(parsed_shape, FloatType, lambda: builder.ir_constant_float(1.0))
        elif parsed_dtype == IntegerType:
            return NDArrayValue.fill(parsed_shape, IntegerType, lambda: builder.ir_constant_int(1))
        else:
            raise TypeInferenceError(dbg, f"Unsupported NDArray dtype {parsed_dtype}")
