from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.op_def.dynamic_ndarray.view_utils import flatten_logical_values


class DynamicNDArray_FlatOp(AbstractOp):
    def get_signature(self) -> str:
        return "DynamicNDArray.flat"

    @classmethod
    def get_name(cls) -> str:
        return "flat"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [AbstractOp._ParamEntry("self")]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, DynamicNDArrayValue)
        values = flatten_logical_values(builder, the_self)
        max_len = the_self.max_length()
        if len(values) < max_len:
            values = values + [builder.ir_constant_int(0) for _ in range(max_len - len(values))]
        return DynamicNDArrayValue.from_max_bounds_and_vector(
            the_self.max_length(),
            the_self.max_rank(),
            the_self.dtype(),
            values[:max_len],
            logical_shape=(max_len,),
            logical_offset=0,
            logical_strides=(1,),
        )
