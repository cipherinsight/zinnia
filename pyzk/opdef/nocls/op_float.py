from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, FloatDTDescriptor, IntegerDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, FloatFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, \
    FloatInferenceDescriptor, IntegerInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class FloatOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "float"

    @classmethod
    def get_name(cls) -> str:
        return "float"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"].type()
        if isinstance(x, IntegerDTDescriptor):
            return FloatDTDescriptor()
        raise TypeInferenceError(dbg_i, f'Invalid float cast on `{x}`, as it must be an Integer')

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, IntegerInferenceDescriptor):
            if x.get() is None:
                return FloatInferenceDescriptor(None)
            return FloatInferenceDescriptor(float(x.get()))
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        return FloatFlattenDescriptor(ir_builder.create_float_cast(x.ptr()))
