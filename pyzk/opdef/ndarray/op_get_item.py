from typing import List, Optional, Dict

from pyzk.debug.dbg_info import DebugInfo
from pyzk.opdef.ndarray.abstract_ndarray_item_slice import AbstractNDArrayItemSlice
from pyzk.opdef.nocls.abstract_op import AbstractOp
from pyzk.builder.abstract_ir_builder import AbsIRBuilderInterface
from pyzk.builder.value import Value, NDArrayValue


class NDArray_GetItemOp(AbstractNDArrayItemSlice):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray::__get_item__"

    @classmethod
    def get_name(cls) -> str:
        return "NDArray::__get_item__"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("slicing_params")
        ]

    def build(self, reducer: AbsIRBuilderInterface, kwargs: Dict[str, Value], dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs['self']
        slicing_params = self.check_slicing_params_datatype(kwargs['slicing_params'], dbg)
        assert isinstance(the_self, NDArrayValue)
        self.check_slicing_dimensions(slicing_params.values(), the_self.shape(), dbg)
        candidates, conditions = self.find_all_candidates(reducer, slicing_params.values(), the_self.shape(), dbg)
        result = the_self.get_item(candidates[0])
        for candidate, condition in zip(candidates[1:], conditions[1:]):
            result = reducer.op_select(condition, the_self.get_item(candidate), result)
        return result
