from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, ClassInferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


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

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        n = kwargs["n"]
        dtype = kwargs["dtype"]
        if not isinstance(n.type(), IntegerDTDescriptor):
            raise TypeInferenceError(dbg_i, "Param `n` must be of type `Number`")
        if n.get() is None:
            raise StaticInferenceError(dbg_i, "Cannot statically infer the value of param `n`")
        if n.get() <= 0:
            raise TypeInferenceError(dbg_i, "Invalid `n` value, n must be greater than 0")
        parsed_dtype = FloatDTDescriptor()
        if dtype is not None and isinstance(dtype, ClassInferenceDescriptor):
            parsed_dtype = dtype.get()
        elif dtype is not None and not isinstance(dtype, ClassInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f"Invalid argument dtype, it must be a datatype")
        if not isinstance(parsed_dtype, FloatDTDescriptor) and not isinstance(parsed_dtype, IntegerDTDescriptor):
            raise TypeInferenceError(dbg_i, f"Unsupported NDArray dtype {parsed_dtype}")
        return NDArrayDTDescriptor((n.get(), n.get()), parsed_dtype)

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        n = kwargs["n"].get()
        dtype = kwargs["dtype"]
        parsed_dtype = FloatDTDescriptor()
        if dtype is not None:
            parsed_dtype = dtype.get()
        ndarray = NDArrayHelper.fill((n, n), lambda: 0)
        ndarray = ndarray.for_each(lambda pos, val: 1 if pos[0] == pos[1] else 0)
        return NDArrayInferenceDescriptor((n, n), parsed_dtype, ndarray)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        n = kwargs["n"]
        dtype = kwargs["dtype"]
        parsed_dtype = FloatDTDescriptor()
        if dtype is not None:
            parsed_dtype = dtype.val()
        constant_0_i = ir_builder.create_constant(0)
        constant_1_i = ir_builder.create_constant(1)
        constant_0_f = ir_builder.create_float_cast(constant_0_i)
        constant_1_f = ir_builder.create_float_cast(constant_1_i)
        if isinstance(parsed_dtype, FloatDTDescriptor):
            ndarray = NDArrayHelper.fill((n.val(), n.val()), lambda: constant_0_f)
            ndarray = ndarray.for_each(lambda pos, val: constant_1_f if pos[0] == pos[1] else constant_0_f)
        elif isinstance(parsed_dtype, IntegerDTDescriptor):
            ndarray = NDArrayHelper.fill((n.val(), n.val()), lambda: constant_0_i)
            ndarray = ndarray.for_each(lambda pos, val: constant_1_i if pos[0] == pos[1] else constant_0_i)
        else:
            raise NotImplementedError()
        return NDArrayFlattenDescriptor((n.val(), n.val()), parsed_dtype, ndarray)
