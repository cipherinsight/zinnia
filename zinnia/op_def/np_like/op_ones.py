from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.type_sys import IntegerType, FloatType, BooleanType
from zinnia.compile.type_sys.ndarray_bounds import infer_ndarray_compile_bounds_from_shape
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, TupleValue, ClassValue, NDArrayValue, NoneValue


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
        is_dynamic = False
        bounds = None
        try:
            bounds = infer_ndarray_compile_bounds_from_shape(shape, builder, dbg, self.get_name())
        except StaticInferenceError:
            is_dynamic = True
        parsed_dtype = FloatType
        if not isinstance(dtype, NoneValue):
            if isinstance(dtype, ClassValue):
                dtype_val = dtype.val(builder)
                if dtype_val not in [FloatType, IntegerType, BooleanType]:
                    raise TypeInferenceError(dbg, f"Unsupported NDArray dtype {dtype_val}")
                parsed_dtype = dtype_val
            else:
                raise TypeInferenceError(dbg, "Invalid type for argument `dtype`, it must be a datatype")
        if is_dynamic:
            return builder.op_dynamic_ndarray_ones(shape, dtype, dbg)

        assert bounds is not None
        parsed_shape = bounds.static_shape

        if parsed_dtype == FloatType:
            return NDArrayValue.fill(parsed_shape, FloatType, lambda: builder.ir_constant_float(1.0))
        elif parsed_dtype == IntegerType:
            return NDArrayValue.fill(parsed_shape, IntegerType, lambda: builder.ir_constant_int(1))
        elif parsed_dtype == BooleanType:
            return NDArrayValue.fill(parsed_shape, BooleanType, lambda: builder.ir_constant_bool(True))
        else:
            raise TypeInferenceError(dbg, f"Unsupported NDArray dtype {parsed_dtype}")
