from typing import Optional

from zinnia.debug.dbg_info import DebugInfo
from zinnia.debug.exception import StaticInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, ListValue, TupleValue, IntegerValue, NoneValue


class AbstractItemSliceOp(AbstractOp):
    def __init__(self):
        super().__init__()

    def check_slicing_params_datatype(self, slicing_params: Value, dbg: Optional[DebugInfo]) -> ListValue:
        if not isinstance(slicing_params, ListValue):
            raise ValueError(f"Internal Error: slicing_params is not a `ListValue`")
        for slicing_param in slicing_params.values():
            if not isinstance(slicing_param, TupleValue) and not isinstance(slicing_param, IntegerValue):
                raise ValueError(f"Internal Error: slicing_param_tuple is not a `TupleValue` or `IntegerValue`")
            if isinstance(slicing_param, TupleValue):
                if len(slicing_param.values()) != 3:
                    raise ValueError(f"Internal Error: Unexpected tuple length {len(slicing_param.values())}")
                if not all(isinstance(x, IntegerValue) or isinstance(x, NoneValue) for x in slicing_param.values()):
                    raise ValueError(f"Internal Error: Unexpected tuple elements type")
                for i in range(3):
                    if isinstance(slicing_param.values()[i], IntegerValue) and slicing_param.values()[i].val() is None:
                        raise StaticInferenceError(dbg, f"Cannot statically infer the values to slicing parameters at compile time when doing range slicing. This makes the result data type of the slicing operation non-deterministic.")
                if isinstance(slicing_param.values()[2], IntegerValue) and slicing_param.values()[2].val() == 0:
                    raise StaticInferenceError(dbg, f"Slice step cannot be 0")
        return slicing_params

    def check_single_slicing_number(self, number: IntegerValue, dim: int, dbg: Optional[DebugInfo] = None):
        if number.val() is not None:
            actual_number = number.val()
            if actual_number < 0:
                actual_number += dim
            if actual_number < 0 or actual_number >= dim:
                raise StaticInferenceError(dbg, f"Slicing Index out of range, expected {0} <= index < {dim}, but got {number.val()}")

    def insert_slicing_number_assertion(self, number: IntegerValue, dim: int, builder: IRBuilderInterface):
        is_neg = builder.ir_less_than_i(number, builder.ir_constant_int(0))
        number = builder.ir_add_i(number, builder.ir_mul_i(builder.ir_constant_int(dim), is_neg))
        is_out_of_range = builder.ir_logical_or(
            builder.ir_less_than_i(number, builder.ir_constant_int(0)),
            builder.ir_logical_not(builder.ir_less_than_i(number, builder.ir_constant_int(dim)))
        )
        builder.op_assert(builder.ir_logical_not(is_out_of_range), builder.op_constant_none())
