from typing import List, Dict, Optional

from pyzk.exception.contextual import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, TupleDTDescriptor
from pyzk.util.flatten_descriptor import NDArrayFlattenDescriptor, TupleFlattenDescriptor, FlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, TupleInferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class ListOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "list"

    @classmethod
    def get_name(cls) -> str:
        return "list"

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("x")
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"].type()
        if isinstance(x, NDArrayDTDescriptor):
            return NDArrayDTDescriptor(x.shape)
        elif isinstance(x, TupleDTDescriptor):
            return NDArrayDTDescriptor((x.length, ))
        raise TypeInferenceError(spi, "`list` operator, which aims converts the param into NDArray, can only be used on `Tuple` or `NDArray`")

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, NDArrayInferenceDescriptor):
            return NDArrayInferenceDescriptor(x.shape(), x.get())
        elif isinstance(x, TupleInferenceDescriptor):
            return NDArrayInferenceDescriptor((x.length(), ), NDArrayHelper((x.length(), ), list(x.get())))
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        if isinstance(x, NDArrayFlattenDescriptor):
            return NDArrayFlattenDescriptor(x.shape(), x.ptr())
        elif isinstance(x, TupleFlattenDescriptor):
            return NDArrayFlattenDescriptor((x.length(), ), NDArrayHelper((x.length(), ), list(x.ptr())))
        raise NotImplementedError()
