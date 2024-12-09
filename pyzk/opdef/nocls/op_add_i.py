from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class AddIOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "add_i"

    @classmethod
    def get_name(cls) -> str:
        return "add_i"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("lhs"),
            AbstractOp._ParamEntry("rhs"),
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        lhs, rhs = kwargs["lhs"].type(), kwargs["rhs"].type()
        if isinstance(lhs, IntegerDTDescriptor) and isinstance(rhs, IntegerDTDescriptor):
            return IntegerDTDescriptor()
        raise TypeInferenceError(dbg_i, f"Operator `{self.get_name()}` only accepts `Integer`")

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, IntegerInferenceDescriptor) and isinstance(rhs, IntegerInferenceDescriptor):
            if lhs.get() is None or rhs.get() is None:
                return IntegerInferenceDescriptor(None)
            return IntegerInferenceDescriptor(lhs.get() + rhs.get())
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        lhs, rhs = kwargs["lhs"], kwargs["rhs"]
        if isinstance(lhs, IntegerFlattenDescriptor) and isinstance(rhs, IntegerFlattenDescriptor):
            return IntegerFlattenDescriptor(ir_builder.create_add_i(lhs.ptr(), rhs.ptr()))
        raise NotImplementedError()
