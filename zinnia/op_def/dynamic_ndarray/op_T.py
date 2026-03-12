from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.op_def.dynamic_ndarray.op_transpose import DynamicNDArray_TransposeOp


class DynamicNDArray_TOp(AbstractOp):
    def get_signature(self) -> str:
        return "DynamicNDArray.T"

    @classmethod
    def get_name(cls) -> str:
        return "T"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [AbstractOp._ParamEntry("self")]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        assert isinstance(the_self, DynamicNDArrayValue)
        op = DynamicNDArray_TransposeOp()
        kwargs2 = op.argparse(dbg, [the_self], {})
        return op.build(builder, OpArgsContainer(kwargs2), dbg)
