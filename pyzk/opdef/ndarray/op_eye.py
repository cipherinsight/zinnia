from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError, StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, ClassInferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


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
            AbstractOp._ParamEntry("dtype", False),
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        n, m = kwargs["n"], kwargs["m"]
        dtype = kwargs["dtype"]
        if not isinstance(n.type(), IntegerDTDescriptor):
            raise TypeInferenceError(dbg_i, "Param `n` must be of type `Number`")
        if not isinstance(m.type(), IntegerDTDescriptor):
            raise TypeInferenceError(dbg_i, "Param `m` must be of type `Number`")
        if n.get() is None:
            raise StaticInferenceError(dbg_i, "Cannot statically infer the value of param `n`")
        if m.get() is None:
            raise StaticInferenceError(dbg_i, "Cannot statically infer the value of param `m`")
        if n.get() <= 0:
            raise TypeInferenceError(dbg_i, "Invalid `n` value, n must be greater than 0")
        if m.get() <= 0:
            raise TypeInferenceError(dbg_i, "Invalid `m` value, m must be greater than 0")
        parsed_dtype = FloatDTDescriptor()
        if dtype is not None and isinstance(dtype, ClassInferenceDescriptor):
            parsed_dtype = dtype.get()
        elif dtype is not None and not isinstance(dtype, ClassInferenceDescriptor):
            raise TypeInferenceError(dbg_i, f"Invalid argument dtype, it must be a datatype")
        if not isinstance(parsed_dtype, FloatDTDescriptor) and not isinstance(parsed_dtype, IntegerDTDescriptor):
            raise TypeInferenceError(dbg_i, f"Unsupported NDArray dtype {parsed_dtype}")
        return NDArrayDTDescriptor((n.get(), m.get()), parsed_dtype)

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        n, m = kwargs["n"].get(), kwargs["m"].get()
        dtype = kwargs["dtype"]
        parsed_dtype = FloatDTDescriptor()
        if dtype is not None:
            parsed_dtype = dtype.get()
        ndarray = NDArrayHelper.fill((n, m), lambda: 0)
        ndarray = ndarray.for_each(lambda pos, val: 1 if pos[0] == pos[1] else 0)
        return NDArrayInferenceDescriptor((n, m), parsed_dtype, ndarray)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        n, m = kwargs["n"], kwargs["m"]
        dtype = kwargs["dtype"]
        parsed_dtype = FloatDTDescriptor()
        if dtype is not None:
            parsed_dtype = dtype.val()
        constant_0_i = ir_builder.create_constant(0)
        constant_1_i = ir_builder.create_constant(1)
        constant_0_f = ir_builder.create_float_cast(constant_0_i)
        constant_1_f = ir_builder.create_float_cast(constant_1_i)
        if isinstance(parsed_dtype, FloatDTDescriptor):
            ndarray = NDArrayHelper.fill((n.val(), m.val()), lambda: constant_0_f)
            ndarray = ndarray.for_each(lambda pos, val: constant_1_f if pos[0] == pos[1] else constant_0_f)
        elif isinstance(parsed_dtype, IntegerDTDescriptor):
            ndarray = NDArrayHelper.fill((n.val(), m.val()), lambda: constant_0_i)
            ndarray = ndarray.for_each(lambda pos, val: constant_1_i if pos[0] == pos[1] else constant_0_i)
        else:
            raise NotImplementedError()
        return NDArrayFlattenDescriptor((n.val(), m.val()), parsed_dtype, ndarray)
