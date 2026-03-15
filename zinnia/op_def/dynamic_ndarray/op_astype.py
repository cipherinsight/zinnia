from typing import List, Optional

from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.compile.triplet import ClassValue, DynamicNDArrayValue, Value
from zinnia.compile.type_sys import FloatType, IntegerType
from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import TypeInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp


class DynamicNDArray_AsTypeOp(AbstractOp):
    def get_signature(self) -> str:
        return "DynamicNDArray.astype"

    @classmethod
    def get_name(cls) -> str:
        return "astype"

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("dtype"),
        ]

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs["self"]
        the_dtype = kwargs["dtype"]
        assert isinstance(the_self, DynamicNDArrayValue)
        if not isinstance(the_dtype, ClassValue):
            raise TypeInferenceError(dbg, f"`dtype` expected a class value, got {the_dtype.type()}")

        target_dtype = the_dtype.val(builder)
        values = the_self.flattened_values()
        if target_dtype == IntegerType:
            casted = [builder.ir_int_cast(x, dbg) for x in values]
        elif target_dtype == FloatType:
            casted = [builder.ir_float_cast(x, dbg) for x in values]
        else:
            raise TypeInferenceError(dbg, f"Unsupported dtype got: {target_dtype}")

        return DynamicNDArrayValue.from_max_bounds_and_vector(
            the_self.max_length(),
            the_self.max_rank(),
            target_dtype,
            casted,
            logical_shape=the_self.logical_shape(),
            logical_offset=the_self.logical_offset(),
            logical_strides=the_self.logical_strides(),
        )
