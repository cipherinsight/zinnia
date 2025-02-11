import copy
from typing import List, Dict, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.type_sys import IntegerType, FloatType
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import NDArrayValue, Value, ClassValue


class NDArray_AsTypeOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def get_signature(self) -> str:
        return "NDArray.astype"

    @classmethod
    def get_name(cls) -> str:
        return "astype"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("dtype")
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        the_dtype = kwargs["dtype"]
        assert isinstance(the_self, NDArrayValue)
        if not isinstance(the_dtype, ClassValue):
            raise TypeInferenceError(dbg, f"`dtype` expected a class value, got {the_dtype.type()}")
        if the_dtype.val() == IntegerType:
            if the_self.dtype() == IntegerType:
                return copy.deepcopy(the_self)
            return the_self.unary(IntegerType, lambda x: builder.ir_int_cast(x, dbg))
        elif the_dtype.val() == FloatType:
            if the_self.dtype() == FloatType:
                return copy.deepcopy(the_self)
            return the_self.unary(FloatType, lambda x: builder.ir_float_cast(x, dbg))
        raise TypeInferenceError(dbg, f"Unsupported dtype got: {the_dtype.val()}")
