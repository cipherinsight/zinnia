from typing import List, Dict, Optional

from zenopy.algo.ndarray_helper import NDArrayValueWrapper
from zenopy.opdef.nocls.abstract_op import AbstractOp
from zenopy.debug.dbg_info import DebugInfo
from zenopy.builder.abstract_ir_builder import AbsIRBuilderInterface
from zenopy.builder.value import Value, NDArrayValue


class NDArray_TOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::T"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::T"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, NDArrayValue)
        new_shape = the_self.shape()[::-1]
        flattened_values = the_self.get().flatten()
        new_values = NDArrayValueWrapper.from_1d_values_and_shape(flattened_values, new_shape)
        return NDArrayValue(new_shape, the_self.dtype(), new_values)
