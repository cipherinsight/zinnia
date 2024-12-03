from typing import List, Dict, Optional

from pyzk.exception.contextual import TypeInferenceError, StaticInferenceError
from pyzk.opdef.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class NDArray_IdentityOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::identity"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::identity"

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("n")
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        n = kwargs["n"]
        if not isinstance(n.type(), NumberDTDescriptor):
            raise TypeInferenceError(spi, "Param `n` must be of type `Number`")
        if n.get() is None:
            raise StaticInferenceError(spi, "Cannot statically infer the value of param `n`")
        if n.get() <= 0:
            raise TypeInferenceError(spi, "Invalid `n` value, n must be greater than 0")
        return NDArrayDTDescriptor((n.get(), n.get()))

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        n = kwargs["n"].get()
        ndarray = NDArrayHelper.fill((n, n), lambda: 0)
        ndarray = ndarray.for_each(lambda pos, val: 1 if pos[0] == pos[1] else 0)
        return NDArrayInferenceDescriptor((n, n), ndarray)
