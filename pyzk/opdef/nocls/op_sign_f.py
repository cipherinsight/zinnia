from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor, FloatFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor, FloatInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class SignFOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "sign_f"

    @classmethod
    def get_name(cls) -> str:
        return "sign_f"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"].type()
        if isinstance(x, FloatDTDescriptor):
            return IntegerDTDescriptor()
        raise TypeInferenceError(dbg_i, f'Invalid logical operator `{self.get_signature()}` on operand {x}, as it must be a Float')

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, FloatInferenceDescriptor):
            if x.get() is None:
                return IntegerInferenceDescriptor(None)
            return IntegerInferenceDescriptor((1 if x.get() > 0 else (0 if x.get() == 0 else -1)) if x is not None else None)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        if isinstance(x, FloatFlattenDescriptor):
            return IntegerFlattenDescriptor(ir_builder.create_sign_f(x.ptr()))
        raise NotImplementedError()
