from typing import List, Dict, Optional

from pyzk.exception.contextual import StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NumberFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor
from pyzk.util.source_pos_info import SourcePosInfo


class ConstantCastOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "constant_cast"

    @classmethod
    def get_name(cls) -> str:
        return "constant_cast"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"]
        if isinstance(x.type(), NumberDTDescriptor):
            if x.get() is None:
                raise StaticInferenceError(spi, 'Cannot statically infer the value')
            return NumberDTDescriptor()
        raise NotImplementedError()

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, NumberInferenceDescriptor):
            return NumberInferenceDescriptor(x.get())
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        return NumberFlattenDescriptor(x.ptr())
