from typing import List, Dict, Any, Optional

from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, NDArrayDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, IntegerFlattenDescriptor, NDArrayFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.debug.dbg_info import DebugInfo


class USubOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "usub"

    @classmethod
    def get_name(cls) -> str:
        return "usub"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("x")
        ]

    def perform_inference(self, lhs: Any, rhs: Any) -> Any:
        raise NotImplementedError()

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        x = kwargs["x"].type()
        if isinstance(x, IntegerDTDescriptor):
            return IntegerDTDescriptor()
        elif isinstance(x, NDArrayDTDescriptor):
            return NDArrayDTDescriptor(x.shape)
        raise NotImplementedError()

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        x = kwargs["x"]
        if isinstance(x, IntegerInferenceDescriptor):
            if x.get() is None:
                return IntegerInferenceDescriptor(None)
            return IntegerInferenceDescriptor(x.get() * -1)
        elif isinstance(x, NDArrayInferenceDescriptor):
            return NDArrayInferenceDescriptor(x.shape(), x.get().unary(lambda a: -a if a is not None else None))
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        x = kwargs["x"]
        minus_one = ir_builder.create_constant(-1)
        if isinstance(x, IntegerFlattenDescriptor):
            return IntegerFlattenDescriptor(ir_builder.create_mul(
                minus_one, x.ptr()
            ))
        elif isinstance(x, NDArrayFlattenDescriptor):
            return NDArrayFlattenDescriptor(x.shape(), x.ptr().unary(lambda a: ir_builder.create_mul(
                minus_one, a
            )))
        raise NotImplementedError()
