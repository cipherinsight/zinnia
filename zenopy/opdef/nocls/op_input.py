from typing import List, Dict, Optional

from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.internal.dt_descriptor import DTDescriptor, IntegerDTDescriptor, NDArrayDTDescriptor, FloatDTDescriptor
from zenopy.algo.ndarray_helper import NDArrayValueWrapper
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, NDArrayValue


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

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return []

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        if isinstance(self.dt, NDArrayDTDescriptor):
            the_idx = 0
            dtype = self.dt.dtype
            def _id_yield():
                nonlocal the_idx
                the_idx += 1
                if isinstance(dtype, IntegerDTDescriptor):
                    return reducer.ir_read_integer(self.input_id, the_idx - 1)
                elif isinstance(dtype, FloatDTDescriptor):
                    return reducer.ir_read_float(self.input_id, the_idx - 1)
                raise NotImplementedError()
            return NDArrayValue(self.dt.shape, dtype, NDArrayValueWrapper.fill(
                self.dt.shape, _id_yield
            ))
        elif isinstance(self.dt, IntegerDTDescriptor):
            return reducer.ir_read_integer(self.input_id, 0)
        elif isinstance(self.dt, FloatDTDescriptor):
            return reducer.ir_read_float(self.input_id, 0)
        raise NotImplementedError()
