from typing import List, Optional, Dict

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_ndarray_item_slice import AbstractNDArrayItemSlice
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue


class NDArray_GetItemOp(AbstractNDArrayItemSlice):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.__get_item__"

    @classmethod
    def get_name(cls) -> str:
        return "__get_item__"

    @classmethod
    def requires_condition(cls) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("slicing_params")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs['self']
        slicing_params = self.check_slicing_params_datatype(builder, kwargs['slicing_params'], dbg)
        assert isinstance(the_self, NDArrayValue)
        self.check_slicing_dimensions(slicing_params.values(), the_self.shape(), dbg)
        candidates, conditions = self.find_all_candidates(builder, slicing_params.values(), the_self.shape(), kwargs.get_condition(), dbg)
        result = the_self.get_item(candidates[0])
        for candidate, condition in zip(candidates[1:], conditions[1:]):
            result = builder.op_select(condition, the_self.get_item(candidate), result)
        return result
