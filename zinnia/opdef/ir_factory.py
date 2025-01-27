from inspect import isclass

from zinnia.opdef.ir_op.abstract_ir import AbstractIR
from zinnia.opdef.ir_op.ir_abs_f import AbsFIR
from zinnia.opdef.ir_op.ir_abs_i import AbsIIR
from zinnia.opdef.ir_op.ir_add_f import AddFIR
from zinnia.opdef.ir_op.ir_add_i import AddIIR
from zinnia.opdef.ir_op.ir_assert import AssertIR
from zinnia.opdef.ir_op.ir_bool_cast import BoolCastIR
from zinnia.opdef.ir_op.ir_constant_float import ConstantFloatIR
from zinnia.opdef.ir_op.ir_constant_int import ConstantIntIR
from zinnia.opdef.ir_op.ir_cos_f import CosFIR
from zinnia.opdef.ir_op.ir_cosh_f import CosHFIR
from zinnia.opdef.ir_op.ir_div_f import DivFIR
from zinnia.opdef.ir_op.ir_div_i import DivIIR
from zinnia.opdef.ir_op.ir_eq_f import EqualFIR
from zinnia.opdef.ir_op.ir_eq_hash import EqualHashIR
from zinnia.opdef.ir_op.ir_eq_i import EqualIIR
from zinnia.opdef.ir_op.ir_exp_f import ExpFIR
from zinnia.opdef.ir_op.ir_export_external_f import ExportExternalFIR
from zinnia.opdef.ir_op.ir_export_external_i import ExportExternalIIR
from zinnia.opdef.ir_op.ir_expose_public_f import ExposePublicFIR
from zinnia.opdef.ir_op.ir_expose_public_i import ExposePublicIIR
from zinnia.opdef.ir_op.ir_float_cast import FloatCastIR
from zinnia.opdef.ir_op.ir_floor_divide_f import FloorDivFIR
from zinnia.opdef.ir_op.ir_floor_divide_i import FloorDivIIR
from zinnia.opdef.ir_op.ir_gt_f import GreaterThanFIR
from zinnia.opdef.ir_op.ir_gt_i import GreaterThanIIR
from zinnia.opdef.ir_op.ir_gte_f import GreaterThanOrEqualFIR
from zinnia.opdef.ir_op.ir_gte_i import GreaterThanOrEqualIIR
from zinnia.opdef.ir_op.ir_poseidon_hash import PoseidonHashIR
from zinnia.opdef.ir_op.ir_int_cast import IntCastIR
from zinnia.opdef.ir_op.ir_invoke_external import InvokeExternalIR
from zinnia.opdef.ir_op.ir_log_f import LogFIR
from zinnia.opdef.ir_op.ir_logical_and import LogicalAndIR
from zinnia.opdef.ir_op.ir_logical_not import LogicalNotIR
from zinnia.opdef.ir_op.ir_logical_or import LogicalOrIR
from zinnia.opdef.ir_op.ir_lt_f import LessThanFIR
from zinnia.opdef.ir_op.ir_lt_i import LessThanIIR
from zinnia.opdef.ir_op.ir_lte_f import LessThanOrEqualFIR
from zinnia.opdef.ir_op.ir_lte_i import LessThanOrEqualIIR
from zinnia.opdef.ir_op.ir_mod_f import ModFIR
from zinnia.opdef.ir_op.ir_mod_i import ModIIR
from zinnia.opdef.ir_op.ir_mul_f import MulFIR
from zinnia.opdef.ir_op.ir_mul_i import MulIIR
from zinnia.opdef.ir_op.ir_ne_f import NotEqualFIR
from zinnia.opdef.ir_op.ir_ne_i import NotEqualIIR
from zinnia.opdef.ir_op.ir_pow_f import PowFIR
from zinnia.opdef.ir_op.ir_pow_i import PowIIR
from zinnia.opdef.ir_op.ir_read_float import ReadFloatIR
from zinnia.opdef.ir_op.ir_read_hash import ReadHashIR
from zinnia.opdef.ir_op.ir_read_integer import ReadIntegerIR
from zinnia.opdef.ir_op.ir_select_f import SelectFIR
from zinnia.opdef.ir_op.ir_select_i import SelectIIR
from zinnia.opdef.ir_op.ir_sign_f import SignFIR
from zinnia.opdef.ir_op.ir_sign_i import SignIIR
from zinnia.opdef.ir_op.ir_sin_f import SinFIR
from zinnia.opdef.ir_op.ir_sinh_f import SinHFIR
from zinnia.opdef.ir_op.ir_sqrt_f import SqrtFIR
from zinnia.opdef.ir_op.ir_sub_f import SubFIR
from zinnia.opdef.ir_op.ir_sub_i import SubIIR
from zinnia.opdef.ir_op.ir_tan_f import TanFIR
from zinnia.opdef.ir_op.ir_tanh_f import TanHFIR


class IRFactory:
    class Registry:
        ABS_F = AbsFIR
        ABS_I = AbsIIR
        ADD_F = AddFIR
        ADD_I = AddIIR
        ASSERT = AssertIR
        BOOL_CAST = BoolCastIR
        CONSTANT_FLOAT = ConstantFloatIR
        CONSTANT_INT = ConstantIntIR
        COS_F = CosFIR
        COSH_F = CosHFIR
        DIV_F = DivFIR
        DIV_I = DivIIR
        EQ_F = EqualFIR
        EQ_I = EqualIIR
        EQ_HASH = EqualHashIR
        EXP_F = ExpFIR
        EXPORT_EXTERNAL_F = ExportExternalFIR
        EXPORT_EXTERNAL_I = ExportExternalIIR
        EXPOSE_PUBLIC_F = ExposePublicFIR
        EXPOSE_PUBLIC_I = ExposePublicIIR
        FLOAT_CAST = FloatCastIR
        FLOOR_DIV_F = FloorDivFIR
        FLOOR_DIV_I = FloorDivIIR
        GT_F = GreaterThanFIR
        GT_I = GreaterThanIIR
        GTE_F = GreaterThanOrEqualFIR
        GTE_I = GreaterThanOrEqualIIR
        HASH = PoseidonHashIR
        INT_CAST = IntCastIR
        INVOKE_EXTERNAL = InvokeExternalIR
        LOG_F = LogFIR
        LOGICAL_AND = LogicalAndIR
        LOGICAL_NOT = LogicalNotIR
        LOGICAL_OR = LogicalOrIR
        LT_F = LessThanFIR
        LT_I = LessThanIIR
        LTE_F = LessThanOrEqualFIR
        LTE_I = LessThanOrEqualIIR
        MOD_F = ModFIR
        MOD_I = ModIIR
        MUL_F = MulFIR
        MUL_I = MulIIR
        NE_F = NotEqualFIR
        NE_I = NotEqualIIR
        POW_F = PowFIR
        POW_I = PowIIR
        READ_FLOAT = ReadFloatIR
        READ_HASH = ReadHashIR
        READ_INT = ReadIntegerIR
        SELECT_F = SelectFIR
        SELECT_I = SelectIIR
        SIGN_F = SignFIR
        SIGN_I = SignIIR
        SIN_F = SinFIR
        SINH_F = SinHFIR
        SQRT_F = SqrtFIR
        SUB_F = SubFIR
        SUB_I = SubIIR
        TAN_F = TanFIR
        TANH_F = TanHFIR

    @staticmethod
    def get_ir_class(ir_class_name: str):
        for k, op in IRFactory.Registry.__dict__.items():
            if not isclass(op) or not issubclass(op, AbstractIR):
                continue
            if op.__name__ == ir_class_name:
                return op
        return None

    @staticmethod
    def export(ir: AbstractIR):
        return {
            "__class__": ir.__class__.__name__,
            "ir_data": ir.export(),
        }

    @staticmethod
    def import_from(data: dict):
        ir_class_name = data["__class__"]
        ir_class = IRFactory.get_ir_class(ir_class_name)
        if ir_class is None:
            raise NotImplementedError(f"Internal Error: IR class {ir_class_name} not found. Please check IR Registry.")
        return ir_class.import_from(data["ir_data"])
