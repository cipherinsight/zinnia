from typing import List, Dict, Optional

from pyzk.exception.contextual import TypeInferenceError, StaticInferenceError
from pyzk.opdef.abstract_op import AbstractOp, _ParamEntry
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


class NDArray_EyeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::eye"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::eye"

    def get_param_entries(self) -> List[_ParamEntry]:
        return [
            _ParamEntry("n"),
            _ParamEntry("m"),
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        n, m = kwargs["n"], kwargs["m"]
        if not isinstance(n.type(), NumberDTDescriptor):
            raise TypeInferenceError(spi, "Param `n` must be of type `Number`")
        if not isinstance(m.type(), NumberDTDescriptor):
            raise TypeInferenceError(spi, "Param `m` must be of type `Number`")
        if n.get() is None:
            raise StaticInferenceError(spi, "Cannot statically infer the value of param `n`")
        if m.get() is None:
            raise StaticInferenceError(spi, "Cannot statically infer the value of param `m`")
        if n.get() <= 0:
            raise TypeInferenceError(spi, "Invalid `n` value, n must be greater than 0")
        if m.get() <= 0:
            raise TypeInferenceError(spi, "Invalid `m` value, m must be greater than 0")
        return NDArrayDTDescriptor((n.get(), m.get()))

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        n, m = kwargs["n"].get(), kwargs["m"].get()
        ndarray = NDArrayHelper.fill((n, m), lambda: 0)
        ndarray = ndarray.for_each(lambda pos, val: 1 if pos[0] == pos[1] else 0)
        return NDArrayInferenceDescriptor((n, m), ndarray)

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        n, m = kwargs["n"], kwargs["m"]
        constant_0 = ir_builder.create_constant(0)
        constant_1 = ir_builder.create_constant(1)
        ndarray = NDArrayHelper.fill((n.val(), m.val()), lambda: constant_0)
        ndarray = ndarray.for_each(lambda pos, val: constant_1 if pos[0] == pos[1] else constant_0)
        return NDArrayFlattenDescriptor((n.val(), m.val()), ndarray)
