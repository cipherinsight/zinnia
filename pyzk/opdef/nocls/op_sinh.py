from typing import List, Dict, Optional

from pyzk.debug.exception import TypeInferenceError
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, FloatDTDescriptor, NDArrayDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor, FloatFlattenDescriptor, \
    NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor, \
    FloatInferenceDescriptor, NDArrayInferenceDescriptor, NDArrayInferenceValue
from pyzk.debug.dbg_info import DebugInfo


class SinHOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "sinh"

    @classmethod
    def get_name(cls) -> str:
        return "sinh"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"].type()
        if isinstance(x, IntegerDTDescriptor):
            return FloatDTDescriptor()
        elif isinstance(x, FloatDTDescriptor):
            return FloatDTDescriptor()
        elif isinstance(x, NDArrayDTDescriptor):
            return NDArrayDTDescriptor(x.shape, FloatDTDescriptor())
        raise TypeInferenceError(dbg_i, f'Operator `{self.get_signature()}` on operand {x} not supported')

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, IntegerInferenceDescriptor):
            return FloatInferenceDescriptor(None)
        elif isinstance(x, FloatInferenceDescriptor):
            return FloatInferenceDescriptor(None)
        elif isinstance(x, NDArrayInferenceDescriptor):
            return NDArrayInferenceDescriptor(x.shape(), FloatDTDescriptor(), NDArrayInferenceValue.fill(x.shape(), lambda: None))
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        if isinstance(x, IntegerFlattenDescriptor):
            return FloatFlattenDescriptor(ir_builder.create_sinh(ir_builder.create_float_cast(x.ptr())))
        elif isinstance(x, FloatFlattenDescriptor):
            return FloatFlattenDescriptor(ir_builder.create_sinh(x.ptr()))
        if isinstance(x, NDArrayFlattenDescriptor):
            if isinstance(x.dtype(), IntegerDTDescriptor):
                return NDArrayFlattenDescriptor(x.shape(), FloatDTDescriptor(), x.ptr().unary(lambda u: ir_builder.create_sinh(ir_builder.create_float_cast(u))))
            elif isinstance(x.dtype(), FloatDTDescriptor):
                return NDArrayFlattenDescriptor(x.shape(), FloatDTDescriptor(), x.ptr().unary(lambda u: ir_builder.create_sinh(u)))
        raise NotImplementedError()
