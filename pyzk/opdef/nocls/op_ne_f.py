from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor, FloatFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor, FloatInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class NotEqualFOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "ne_f"

    @classmethod
    def get_name(cls) -> str:
        return "ne_f"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["lhs"].type(), kwargs["rhs"].type()
        if isinstance(lhs, FloatDTDescriptor) and isinstance(rhs, FloatDTDescriptor):
            return IntegerDTDescriptor()
        raise TypeInferenceError(dbg_i, f"Operator `{self.get_name()}` only accepts `Float`")

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, FloatInferenceDescriptor) and isinstance(rhs, FloatInferenceDescriptor):
            return IntegerInferenceDescriptor(None)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, FloatFlattenDescriptor) and isinstance(rhs, FloatFlattenDescriptor):
            return IntegerFlattenDescriptor(ir_builder.create_not_equal_f(lhs.ptr(), rhs.ptr()))
        raise NotImplementedError()
