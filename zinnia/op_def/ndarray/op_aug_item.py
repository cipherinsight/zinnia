from typing import List

from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value
from zinnia.op_def.ndarray.op_set_item import NDArray_SetItemOp


class NDArray_AugItemOp(NDArray_SetItemOp):
    class AugOpName:
        ADD = "add"
        SUB = "sub"
        MUL = "mul"
        DIV = "div"
        MOD = "mod"
        POW = "pow"
        FLOOR_DIV = "floor_div"
        MAT_MUL = "mat_mul"

    def __init__(self, op_name: str):
        super().__init__()
        self.op_name = op_name

    def get_signature(self) -> str:
        return "NDArray.__aug_item__"

    @classmethod
    def get_name(cls) -> str:
        return "__aug_item__"

    @classmethod
    def is_inplace(cls) -> bool:
        return True

    def get_param_entries(self) -> List[AbstractOp._ParamEntry]:
        return [
            AbstractOp._ParamEntry("self"),
            AbstractOp._ParamEntry("value"),
            AbstractOp._ParamEntry("slicing_params")
        ]

    def augment_operation(self, builder: IRBuilderInterface, assignee: Value, new_value: Value) -> Value:
        if self.op_name == self.AugOpName.ADD:
            return builder.op_add(assignee, new_value)
        if self.op_name == self.AugOpName.SUB:
            return builder.op_subtract(assignee, new_value)
        if self.op_name == self.AugOpName.MUL:
            return builder.op_multiply(assignee, new_value)
        if self.op_name == self.AugOpName.DIV:
            return builder.op_divide(assignee, new_value)
        if self.op_name == self.AugOpName.FLOOR_DIV:
            return builder.op_floor_divide(assignee, new_value)
        if self.op_name == self.AugOpName.MOD:
            return builder.op_modulo(assignee, new_value)
        if self.op_name == self.AugOpName.POW:
            return builder.op_power(assignee, new_value)
        if self.op_name == self.AugOpName.MAT_MUL:
            return builder.op_mat_mul(assignee, new_value)
        raise NotImplementedError(f"AugItemOp {self.op_name} not implemented")
