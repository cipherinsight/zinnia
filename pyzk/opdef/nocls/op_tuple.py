from typing import List, Dict, Optional

from pyzk.exception.contextual import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NDArrayDTDescriptor, TupleDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor, TupleFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor, TupleInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class TupleOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "tuple"

    @classmethod
    def get_name(cls) -> str:
        return "tuple"

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("x")
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"].type()
        if isinstance(x, NDArrayDTDescriptor):
            if len(x.shape) != 1:
                raise TypeInferenceError(spi, "Cannot cast this `NDArray` to `Tuple`, as its number of dimensions is greater than 1")
            return TupleDTDescriptor(x.shape[0])
        elif isinstance(x, TupleDTDescriptor):
            return TupleDTDescriptor(x.length)
        raise TypeInferenceError(spi, "`tuple` operator, which aims converts the param into Tuple, can only be used on `Tuple` or `NDArray`")

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, NDArrayInferenceDescriptor):
            return TupleInferenceDescriptor(x.shape()[0], tuple(x.get().values))
        elif isinstance(x, TupleInferenceDescriptor):
            return TupleInferenceDescriptor(x.length(), x.get())
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        if isinstance(x, NDArrayFlattenDescriptor):
            return TupleFlattenDescriptor(x.shape()[0], tuple(x.ptr().values))
        elif isinstance(x, TupleFlattenDescriptor):
            return TupleFlattenDescriptor(x.length(), x.ptr())
        raise NotImplementedError()
