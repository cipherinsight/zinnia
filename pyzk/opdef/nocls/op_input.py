from typing import List, Dict, Optional

from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor
from pyzk.internal.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor, IntegerFlattenDescriptor, \
    FloatFlattenDescriptor
from pyzk.internal.inference_descriptor import InferenceDescriptor, IntegerInferenceDescriptor, \
    NDArrayInferenceDescriptor, FloatInferenceDescriptor
from pyzk.algo.ndarray_helper import NDArrayHelper
from pyzk.debug.dbg_info import DebugInfo


class InputOp(AbstractOp):
    def __init__(self, input_id: int, dt: DTDescriptor, public: bool):
        super().__init__()
        self.input_id = input_id
        self.dt = dt
        self.public = public

    def get_signature(self) -> str:
        return "input"

    @classmethod
    def get_name(cls) -> str:
        return "input"

    def dce_keep(self) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def type_check(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        if isinstance(self.dt, NDArrayDTDescriptor):
            return NDArrayDTDescriptor(self.dt.shape, self.dt.dtype)
        elif isinstance(self.dt, IntegerDTDescriptor):
            return IntegerDTDescriptor()
        elif isinstance(self.dt, FloatDTDescriptor):
            return FloatDTDescriptor()
        raise NotImplementedError()

    def static_infer(self, dbg_i: Optional[DebugInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        if isinstance(self.dt, NDArrayDTDescriptor):
            return NDArrayInferenceDescriptor(self.dt.shape, self.dt.dtype, NDArrayHelper.fill(self.dt.shape, lambda: None))
        elif isinstance(self.dt, IntegerDTDescriptor):
            return IntegerInferenceDescriptor(None)
        elif isinstance(self.dt, FloatDTDescriptor):
            return FloatInferenceDescriptor(None)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        if isinstance(self.dt, NDArrayDTDescriptor):
            the_idx = 0
            dtype = self.dt.dtype
            def _id_yield():
                nonlocal the_idx
                the_idx += 1
                if isinstance(dtype, IntegerDTDescriptor):
                    return ir_builder.create_read_integer(self.input_id, the_idx - 1)
                elif isinstance(dtype, FloatDTDescriptor):
                    return ir_builder.create_read_float(self.input_id, the_idx - 1)
                raise NotImplementedError()
            return NDArrayFlattenDescriptor(self.dt.shape, dtype, NDArrayHelper.fill(
                self.dt.shape, _id_yield
            ))
        elif isinstance(self.dt, IntegerDTDescriptor):
            return IntegerFlattenDescriptor(ir_builder.create_read_integer(self.input_id, 0))
        elif isinstance(self.dt, FloatDTDescriptor):
            return FloatFlattenDescriptor(ir_builder.create_read_float(self.input_id, 0))
        raise NotImplementedError()
