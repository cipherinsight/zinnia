from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, IntegerValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_op import AbstractOp


class DynamicNDArray_SizeOp(AbstractOp):
    def get_signature(self) -> str:
        return "DynamicNDArray.size"

    @classmethod
    def get_name(cls) -> str:
        return "size"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [AbstractOp._ParamEntry("self")]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, DynamicNDArrayValue)
        logical_shape = the_self.logical_shape()
        if len(logical_shape) == 1:
            runtime_len = the_self.runtime_logical_length()
            if runtime_len.val(builder) is not None:
                return builder.ir_constant_int(runtime_len.val(builder))
            return runtime_len
        size = 1
        for dim in logical_shape:
            size *= dim
        return builder.ir_constant_int(size)
