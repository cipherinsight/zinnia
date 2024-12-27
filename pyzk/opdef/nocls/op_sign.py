from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor, NDArrayDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor, FloatFlattenDescriptor, \
    NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor, \
    FloatInferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class SignOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "sign"

    @classmethod
    def get_name(cls) -> str:
        return "sign"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"].type()
        if isinstance(x, IntegerDTDescriptor):
            return IntegerDTDescriptor()
        elif isinstance(x, FloatDTDescriptor):
            return IntegerDTDescriptor()
        elif isinstance(x, NDArrayDTDescriptor):
            return NDArrayDTDescriptor(x.shape, IntegerDTDescriptor())
        raise TypeInferenceError(dbg_i, f'Operator `{self.get_signature()}` on operand {x} not supported')

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, IntegerInferenceDescriptor):
            return IntegerInferenceDescriptor((1 if x.get() > 0 else (0 if x.get() == 0 else -1)) if x is not None else None)
        elif isinstance(x, FloatInferenceDescriptor):
            return IntegerInferenceDescriptor((1 if x.get() > 0 else (0 if x.get() == 0 else -1)) if x is not None else None)
        elif isinstance(x, NDArrayInferenceDescriptor):
            return NDArrayInferenceDescriptor(x.shape(), IntegerDTDescriptor(), x.get().unary(
                lambda u: (1 if u.get() > 0 else (0 if u.get() == 0 else -1)) if u is not None else None
            ))
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        if isinstance(x, IntegerFlattenDescriptor):
            return IntegerFlattenDescriptor(ir_builder.create_sign_i(x.ptr()))
        elif isinstance(x, FloatFlattenDescriptor):
            return IntegerFlattenDescriptor(ir_builder.create_sign_f(x.ptr()))
        if isinstance(x, NDArrayFlattenDescriptor):
            if isinstance(x.dtype(), IntegerDTDescriptor):
                return NDArrayFlattenDescriptor(x.shape(), IntegerDTDescriptor(), x.ptr().unary(lambda u: ir_builder.create_sign_i(u)))
            elif isinstance(x.dtype(), FloatDTDescriptor):
                return NDArrayFlattenDescriptor(x.shape(), IntegerDTDescriptor(), x.ptr().unary(lambda u: ir_builder.create_sign_f(u)))
        raise NotImplementedError()
