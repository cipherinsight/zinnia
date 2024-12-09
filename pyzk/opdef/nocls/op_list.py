from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, TupleDTDescriptor, IntegerDTDescriptor
from pyzk.internal.flatten_descriptor import NDArrayFlattenDescriptor, TupleFlattenDescriptor, FlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, TupleInferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


class ListOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "list"

    @classmethod
    def get_name(cls) -> str:
        return "list"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"].type()
        if isinstance(x, NDArrayDTDescriptor):
            return NDArrayDTDescriptor(x.shape, x.dtype)
        elif isinstance(x, TupleDTDescriptor):
            return NDArrayDTDescriptor((x.length, ), IntegerDTDescriptor())
        raise TypeInferenceError(dbg_i, "`list` operator, which aims converts the param into NDArray, can only be used on `Tuple` or `NDArray`")

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, NDArrayInferenceDescriptor):
            return NDArrayInferenceDescriptor(x.shape(), x.dtype(), x.get())
        elif isinstance(x, TupleInferenceDescriptor):
            return NDArrayInferenceDescriptor((x.length(), ), IntegerDTDescriptor(), NDArrayHelper((x.length(), ), list(x.get())))
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        if isinstance(x, NDArrayFlattenDescriptor):
            return NDArrayFlattenDescriptor(x.shape(), x.dtype(), x.ptr())
        elif isinstance(x, TupleFlattenDescriptor):
            return NDArrayFlattenDescriptor((x.length(), ), IntegerDTDescriptor(), NDArrayHelper((x.length(), ), list(x.ptr())))
        raise NotImplementedError()
