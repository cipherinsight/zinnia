from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, FloatDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, FloatFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, FloatInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class PowFOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "pow_f"

    @classmethod
    def get_name(cls) -> str:
        return "pow_f"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x"),
            AbstractOp._ParamEntry("exponent"),
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["x"].type(), kwargs["exponent"].type()
        if isinstance(lhs, FloatDTDescriptor) and isinstance(rhs, FloatDTDescriptor):
            return FloatDTDescriptor()
        raise TypeInferenceError(dbg_i, f"Operator `{self.get_name()}` only accepts `Float`")

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["x"], kwargs["exponent"]
        if isinstance(lhs, FloatInferenceDescriptor) and isinstance(rhs, FloatInferenceDescriptor):
            return FloatInferenceDescriptor(None)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        lhs, rhs = kwargs["x"], kwargs["exponent"]
        if isinstance(lhs, FloatFlattenDescriptor) and isinstance(rhs, FloatFlattenDescriptor):
            return FloatFlattenDescriptor(ir_builder.create_pow_f(lhs.ptr(), rhs.ptr()))
        raise NotImplementedError()
