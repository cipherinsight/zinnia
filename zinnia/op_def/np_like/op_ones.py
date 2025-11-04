from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.type_sys import IntegerType, FloatType, BooleanType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, TupleValue, IntegerValue, ClassValue, NDArrayValue, NoneValue


class NP_OnesOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "np.ones"

    @classmethod
    def get_name(cls) -> str:
        return "ones"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("shape"),
            AbstractOp._ParamEntry("dtype", True)
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        shape = kwargs["shape"]
        dtype = kwargs.get("dtype", builder.op_constant_none())
        if not isinstance(shape, TupleValue):
            raise TypeInferenceError(dbg, "Param `shape` must be of type `Tuple`")
        for ele_t, ele_v in zip(shape.types(), shape.values()):
            if ele_t != IntegerType:
                raise TypeInferenceError(dbg, "Every element in `shape` Tuple must be of type `Integer`")
            assert isinstance(ele_v, IntegerValue)
            if ele_v.val(builder) is None:
                raise StaticInferenceError(dbg, "Every number element in `shape` must be statically inferrable")
            if ele_v.val(builder) <= 0:
                raise TypeInferenceError(dbg, "Every number element in `shape` must be greater than 0")
        parsed_dtype = FloatType
        if not isinstance(dtype, NoneValue):
            if isinstance(dtype, ClassValue):
                parsed_dtype = dtype.val(builder)
            else:
                raise TypeInferenceError(dbg, f"Invalid type for argument `dtype`: {dtype.type()}, it must be a datatype")
        parsed_shape = tuple(v.val(builder) for v in shape.values())
        if parsed_dtype == FloatType:
            return NDArrayValue.fill(parsed_shape, FloatType, lambda: builder.ir_constant_float(1.0))
        elif parsed_dtype == IntegerType:
            return NDArrayValue.fill(parsed_shape, IntegerType, lambda: builder.ir_constant_int(1))
        elif parsed_dtype == BooleanType:
            return NDArrayValue.fill(parsed_shape, BooleanType, lambda: builder.ir_constant_int(1))
        else:
            raise TypeInferenceError(dbg, f"Unsupported NDArray dtype {parsed_dtype}")
