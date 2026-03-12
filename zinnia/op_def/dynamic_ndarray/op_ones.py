from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, TupleValue, ClassValue, NoneValue
from zinnia.compile.type_sys import FloatType, IntegerType, BooleanType
from zinnia.compile.type_sys.ndarray_bounds import infer_ndarray_max_bounds_from_shape
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class DynamicNDArray_OnesOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "DynamicNDArray.ones"

    @classmethod
    def get_name(cls) -> str:
        return "ones"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("shape"),
            AbstractOp._ParamEntry("dtype", True),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> DynamicNDArrayValue:
        shape = kwargs["shape"]
        dtype = kwargs.get("dtype", builder.op_constant_none())
        if not isinstance(shape, TupleValue):
            raise TypeInferenceError(dbg, "Param `shape` must be of type `Tuple`")

        bounds = infer_ndarray_max_bounds_from_shape(shape, builder, dbg, self.get_name())
        parsed_dtype = FloatType
        if not isinstance(dtype, NoneValue):
            if isinstance(dtype, ClassValue):
                parsed_dtype = dtype.val(builder)
            else:
                raise TypeInferenceError(dbg, "Invalid argument `dtype`, it must be a datatype")

        if parsed_dtype == FloatType:
            values = [builder.ir_constant_float(1.0) for _ in range(bounds.max_length)]
        elif parsed_dtype == IntegerType:
            values = [builder.ir_constant_int(1) for _ in range(bounds.max_length)]
        elif parsed_dtype == BooleanType:
            values = [builder.ir_constant_bool(True) for _ in range(bounds.max_length)]
        else:
            raise TypeInferenceError(dbg, f"Unsupported NDArray dtype {parsed_dtype}")

        return DynamicNDArrayValue.from_max_bounds_and_vector(
            bounds.max_length,
            bounds.max_rank,
            parsed_dtype,
            values,
        )