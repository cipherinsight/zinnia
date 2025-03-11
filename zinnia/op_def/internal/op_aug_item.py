from typing import List, Optional

from zinnia.compile.builder.op_args_container import OpArgsContainer
from zinnia.debug.exception import TypeInferenceError, StaticInferenceError
from zinnia.op_def.abstract.abstract_op import AbstractOp
from zinnia.debug.dbg_info import DebugInfo
from zinnia.compile.builder.ir_builder_interface import IRBuilderInterface
from zinnia.compile.triplet import Value, NDArrayValue, ListValue
from zinnia.op_def.internal.op_set_item import SetItemOp


class AugItemOp(SetItemOp):
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
        return f"__aug_item__[{self.op_name}]"

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

    def build(self, builder: IRBuilderInterface, kwargs: OpArgsContainer, dbg: Optional[DebugInfo] = None) -> Value:
        the_self = kwargs['self']
        the_value = kwargs['value']
        slicing_params = self.check_slicing_params_datatype(kwargs['slicing_params'], dbg)
        if isinstance(the_self, ListValue):
            if len(slicing_params.values()) != 1:
                raise StaticInferenceError(dbg, f"List set_item should have exactly one slicing parameter")
            slicing_param = slicing_params.values()[0]
            return self._build_list_assignment(
                builder, kwargs.get_condition(), lambda x, y: self.augment_operation(builder, x, y), the_self, the_value, slicing_param, dbg)
        if isinstance(the_self, NDArrayValue):
            return builder.op_ndarray_aug_item(kwargs.get_condition(), self.op_name, the_self, slicing_params, the_value, dbg)
        raise TypeInferenceError(dbg, f"{the_self.type()} does not support item assignment")
