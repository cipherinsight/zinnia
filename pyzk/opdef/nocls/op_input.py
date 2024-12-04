from typing import List, Dict, Optional

from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.util.dt_descriptor import DTDescriptor, NumberDTDescriptor, NDArrayDTDescriptor
from pyzk.util.flatten_descriptor import FlattenDescriptor, NDArrayFlattenDescriptor, NumberFlattenDescriptor
from pyzk.util.inference_descriptor import InferenceDescriptor, NumberInferenceDescriptor, NDArrayInferenceDescriptor
from pyzk.util.ndarray_helper import NDArrayHelper
from pyzk.util.source_pos_info import SourcePosInfo


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

    def type_check(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> DTDescriptor:
        if isinstance(self.dt, NDArrayDTDescriptor):
            return NDArrayDTDescriptor(self.dt.shape)
        elif isinstance(self.dt, NumberDTDescriptor):
                return NumberDTDescriptor()
        raise NotImplementedError()

    def static_infer(self, spi: Optional[SourcePosInfo], kwargs: Dict[str, InferenceDescriptor]) -> InferenceDescriptor:
        if isinstance(self.dt, NDArrayDTDescriptor):
            return NDArrayInferenceDescriptor(self.dt.shape, NDArrayHelper.fill(self.dt.shape, lambda: None))
        elif isinstance(self.dt, NumberDTDescriptor):
            return NumberInferenceDescriptor(None)
        raise NotImplementedError()

    def ir_flatten(self, ir_builder, kwargs: Dict[str, FlattenDescriptor]) -> FlattenDescriptor:
        if isinstance(self.dt, NDArrayDTDescriptor):
            the_idx = 0
            def _id_yield():
                nonlocal the_idx
                the_idx += 1
                return ir_builder.create_read_number(self.input_id, the_idx - 1)
            return NDArrayFlattenDescriptor(self.dt.shape, NDArrayHelper.fill(
                self.dt.shape, _id_yield
            ))
        elif isinstance(self.dt, NumberDTDescriptor):
            return NumberFlattenDescriptor(ir_builder.create_read_number(self.input_id, 0))
        raise NotImplementedError()
