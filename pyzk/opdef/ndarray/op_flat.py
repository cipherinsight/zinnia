from typing import List, Dict, Optional

from pyzk.algo.ndarray_helper import NDArrayValueWrapper
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.debug.dbg_info import DebugInfo
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, NDArrayValue


class NDArray_FlatOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::flat"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::flat"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, NDArrayValue)
        the_shape = the_self.shape()
        num_items = 1
        for x in the_shape:
            num_items *= x
        flatten_items = the_self.get().flatten()
        return NDArrayValue((num_items, ), the_self.dtype(), NDArrayValueWrapper((num_items, ), flatten_items))
