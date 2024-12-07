from typing import List, Dict, Optional

from pyzk.debug.exception import StaticInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, NumberDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NumberFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


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

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"]
        if isinstance(x.type(), NumberDTDescriptor):
            if x.get() is None:
                raise StaticInferenceError(dbg_i, 'Cannot statically infer the value')
            return NumberDTDescriptor()
        raise NotImplementedError()

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, NumberInferenceDescriptor):
            return NumberInferenceDescriptor(x.get())
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        return NumberFlattenDescriptor(x.ptr())
