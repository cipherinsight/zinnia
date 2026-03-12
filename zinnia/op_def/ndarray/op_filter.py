from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import DynamicNDArrayValue, NDArrayValue, Value
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.op_def.dynamic_ndarray.op_filter import DynamicNDArray_FilterOp


class NDArray_FilterOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.filter"

    @classmethod
    def get_name(cls) -> str:
        return "filter"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("mask"),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        mask = kwargs["mask"]
        if not isinstance(the_self, NDArrayValue):
            raise TypeInferenceError(dbg, "Param `self` must be NDArray")
        if not isinstance(mask, NDArrayValue):
            raise TypeInferenceError(dbg, "Param `mask` must be NDArray")

        dyn_self = the_self if isinstance(the_self, DynamicNDArrayValue) else the_self.to_dynamic_ndarray()

        op = DynamicNDArray_FilterOp()
        kwargs2 = op.argparse(dbg, [dyn_self, mask], {})
        return op.build(builder, OpArgsContainer(kwargs2), dbg)
