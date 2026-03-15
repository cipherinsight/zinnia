from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import ClassValue, DynamicNDArrayValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_op import AbstractOp


class DynamicNDArray_DtypeOp(AbstractOp):
    def get_signature(self) -> str:
        return "DynamicNDArray.dtype"

    @classmethod
    def get_name(cls) -> str:
        return "dtype"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [AbstractOp._ParamEntry("self")]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, DynamicNDArrayValue)
        return ClassValue(the_self.dtype())
