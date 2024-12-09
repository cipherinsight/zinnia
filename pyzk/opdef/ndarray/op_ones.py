from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, TupleDTDescriptor, FloatDTDescriptor, \
    IntegerDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, ClassInferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


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

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        shape = kwargs["shape"]
        dtype = kwargs["dtype"]
        if not isinstance(shape.type(), TupleDTDescriptor):
            raise TypeInferenceError(dbg_i, "Param `shape` must be of type `Tuple`")
        for ele in shape.get():
            if ele is None:
                raise StaticInferenceError(dbg_i, "Every number element in `shape` must be statically inferrable")
            if ele <= 0:
                raise TypeInferenceError(dbg_i, "Every number element in `shape` must be greater than 0")
        parsed_dtype = FloatDTDescriptor()
        if dtype is not None and isinstance(dtype, ClassInferenceDescriptor):
            parsed_dtype = dtype.get()
        elif dtype is not None and not isinstance(dtype, ClassInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f"Invalid argument dtype, it must be a datatype")
        if not isinstance(parsed_dtype, FloatDTDescriptor) and not isinstance(parsed_dtype, IntegerDTDescriptor):
            raise TypeInferenceError(dbg_i, f"Unsupported NDArray dtype {parsed_dtype}")
        return NDArrayDTDescriptor(shape.get(), parsed_dtype)

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        shape = kwargs["shape"].get()
        dtype = kwargs["dtype"]
        parsed_dtype = FloatDTDescriptor()
        if dtype is not None:
            parsed_dtype = dtype.get()
        if isinstance(parsed_dtype, FloatDTDescriptor):
            ndarray = NDArrayHelper.fill(shape, lambda: 1.0)
        else:
            ndarray = NDArrayHelper.fill(shape, lambda: 1)
        return NDArrayInferenceDescriptor(shape, parsed_dtype, ndarray)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        shape = kwargs["shape"]
        dtype = kwargs["dtype"]
        constant_1_i = ir_builder.create_constant(1)
        constant_1_f = ir_builder.create_float_cast(constant_1_i)
        parsed_dtype = FloatDTDescriptor()
        if dtype is not None:
            parsed_dtype = dtype.val()
        if isinstance(parsed_dtype, FloatDTDescriptor):
            ndarray = NDArrayHelper.fill(shape.val(), lambda: constant_1_f)
        else:
            ndarray = NDArrayHelper.fill(shape.val(), lambda: constant_1_i)
        return NDArrayFlattenDescriptor(shape.val(), parsed_dtype, ndarray)
