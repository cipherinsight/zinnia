from typing import List

from zenopy.backend.abstract_builder import AbstractProgramBuilder
from zenopy.backend.zk_program import Halo2ZKProgram
from zenopy.internal.dt_descriptor import IntegerDTDescriptor, FloatDTDescriptor, NDArrayDTDescriptor
from zenopy.internal.prog_meta_data import ProgramMetadata
from zenopy.ir.ir_stmt import IRStatement
from zenopy.opdef.ir_op.ir_abs_f import AbsFIR
from zenopy.opdef.ir_op.ir_abs_i import AbsIIR
from zenopy.opdef.ir_op.ir_add_f import AddFIR
from zenopy.opdef.ir_op.ir_add_i import AddIIR
from zenopy.opdef.ir_op.ir_assert import AssertIR
from zenopy.opdef.ir_op.ir_bool_cast import BoolCastIR
from zenopy.opdef.ir_op.ir_constant_int import ConstantIntIR
from zenopy.opdef.ir_op.ir_constant_float import ConstantFloatIR
from zenopy.opdef.ir_op.ir_cos_f import CosFIR
from zenopy.opdef.ir_op.ir_cosh_f import CosHFIR
from zenopy.opdef.ir_op.ir_div_f import DivFIR
from zenopy.opdef.ir_op.ir_div_i import DivIIR
from zenopy.opdef.ir_op.ir_eq_f import EqualFIR
from zenopy.opdef.ir_op.ir_eq_i import EqualIIR
from zenopy.opdef.ir_op.ir_exp_f import ExpFIR
from zenopy.opdef.ir_op.ir_float_cast import FloatCastIR
from zenopy.opdef.ir_op.ir_floor_divide_f import FloorDivFIR
from zenopy.opdef.ir_op.ir_floor_divide_i import FloorDivIIR
from zenopy.opdef.ir_op.ir_gt_f import GreaterThanFIR
from zenopy.opdef.ir_op.ir_gt_i import GreaterThanIIR
from zenopy.opdef.ir_op.ir_gte_f import GreaterThanOrEqualFIR
from zenopy.opdef.ir_op.ir_gte_i import GreaterThanOrEqualIIR
from zenopy.opdef.ir_op.ir_int_cast import IntCastIR
from zenopy.opdef.ir_op.ir_log_f import LogFIR
from zenopy.opdef.ir_op.ir_logical_or import LogicalOrIR
from zenopy.opdef.ir_op.ir_lt_f import LessThanFIR
from zenopy.opdef.ir_op.ir_lt_i import LessThanIIR
from zenopy.opdef.ir_op.ir_lte_f import LessThanOrEqualFIR
from zenopy.opdef.ir_op.ir_lte_i import LessThanOrEqualIIR
from zenopy.opdef.ir_op.ir_mod_f import ModFIR
from zenopy.opdef.ir_op.ir_mod_i import ModIIR
from zenopy.opdef.ir_op.ir_mul_f import MulFIR
from zenopy.opdef.ir_op.ir_mul_i import MulIIR
from zenopy.opdef.ir_op.ir_ne_f import NotEqualFIR
from zenopy.opdef.ir_op.ir_ne_i import NotEqualIIR
from zenopy.opdef.ir_op.ir_logical_not import LogicalNotIR
from zenopy.opdef.ir_op.ir_logical_and import LogicalAndIR
from zenopy.opdef.ir_op.ir_pow_f import PowFIR
from zenopy.opdef.ir_op.ir_pow_i import PowIIR
from zenopy.opdef.ir_op.ir_read_float import ReadFloatIR
from zenopy.opdef.ir_op.ir_read_integer import ReadIntegerIR
from zenopy.opdef.ir_op.ir_select_f import SelectFIR
from zenopy.opdef.ir_op.ir_select_i import SelectIIR
from zenopy.opdef.ir_op.ir_sign_f import SignFIR
from zenopy.opdef.ir_op.ir_sign_i import SignIIR
from zenopy.opdef.ir_op.ir_sin_f import SinFIR
from zenopy.opdef.ir_op.ir_sinh_f import SinHFIR
from zenopy.opdef.ir_op.ir_sub_f import SubFIR
from zenopy.opdef.ir_op.ir_sub_i import SubIIR
from zenopy.opdef.ir_op.ir_tan_f import TanFIR
from zenopy.opdef.ir_op.ir_tanh_f import TanHFIR


class _Halo2StatementBuilder:
    def __init__(self):
        self.id_var_lookup = {}
        self.id_val_lookup = {}

    def build_stmt(self, stmt: IRStatement) -> str:
        typename = type(stmt.operator).__name__
        method_name = '_build_' + typename
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(method_name)
        return method(stmt)

    def _get_var_name(self, _id: int) -> str:
        var_name = self.id_var_lookup.get(_id, None)
        if var_name is not None:
            return var_name
        var_name = f"y_{_id}"
        self.id_var_lookup[_id] = var_name
        return var_name

    def _build_AddFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, AddFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qadd(ctx, {lhs}, {rhs});"]

    def _build_SubFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, SubFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qsub(ctx, {lhs}, {rhs});"]

    def _build_MulFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, MulFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qmul(ctx, {lhs}, {rhs});"]

    def _build_DivFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, DivFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qdiv(ctx, {lhs}, {rhs});"]

    def _build_AddIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, AddIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.add(ctx, {lhs}, {rhs});"]

    def _build_SubIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, SubIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.sub(ctx, {lhs}, {rhs});"]

    def _build_MulIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, MulIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.mul(ctx, {lhs}, {rhs});"]

    def _build_DivIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, DivIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.div_unsafe(ctx, {lhs}, {rhs});"]

    def _build_AssertIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, AssertIR)
        test = self._get_var_name(stmt.arguments[0])
        return [f"gate.assert_is_const(ctx, &{test}, &F::ONE);"]

    def _build_ReadIntegerIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ReadIntegerIR)
        major: int = stmt.operator.major
        minor: int = stmt.operator.minor
        return [
            f"let tmp_1 = ctx.load_witness(F::from((input.x_{major}_{minor}).abs() as u64));",
            f"let {self._get_var_name(stmt.stmt_id)} = if input.x_{major}_{minor} >= 0 {{tmp_1}} else {{gate.neg(ctx, tmp_1)}};"
        ]

    def _build_ReadFloatIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ReadFloatIR)
        major: int = stmt.operator.major
        minor: int = stmt.operator.minor
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = ctx.load_witness(fixed_point_chip.quantization(input.x_{major}_{minor}));"
        ]

    def _build_ConstantIntIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ConstantIntIR)
        constant_val = stmt.operator.value
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = " + (f"Constant(F::from({constant_val}));" if constant_val >= 0 else f"{{gate.neg(ctx, Constant(F::from({-constant_val})))}};")
        ]

    def _build_ConstantFloatIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ConstantFloatIR)
        constant_val = stmt.operator.value
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = Constant(fixed_point_chip.quantization({constant_val}));"
        ]

    def _build_FloatCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, FloatCastIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let tmp_1 = {x}.value().get_lower_128();",
            f"let tmp_2 = gate.neg(ctx, {x});"
            f"let tmp_3 = tmp_2.value().get_lower_128();",
            f"let tmp_4 = range_chip.is_less_than(ctx, {x}, Constant(F::from(0)), 128);",
            f"let tmp_5 = tmp_4.value().get_lower_128() != 0;",
            f"let tmp_6 = if tmp_5 {{ctx.load_witness(fixed_point_chip.quantization(-(tmp_3 as f64)))}} else {{ctx.load_witness(fixed_point_chip.quantization(tmp_1 as f64))}};",
            f"let tmp_7 = if tmp_5 {{gate.is_equal(ctx, Constant(F::from_u128(tmp_3)), tmp_2)}} else {{gate.is_equal(ctx, Constant(F::from_u128(tmp_1)), {x})}};",
            f"gate.assert_is_const(ctx, &tmp_7, &F::ONE);",
            f"let {self._get_var_name(stmt.stmt_id)} = tmp_6;"
        ]

    def _build_IntCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, IntCastIR)
        x = self._get_var_name(stmt.arguments[0])
        return [f"let {self._get_var_name(stmt.stmt_id)} = if fixed_point_chip.dequantization({x}) >= 0 {{Constant(F::from(fixed_point_chip.dequantization({x}) as u64))}} else {{gate.neg(ctx, Constant(F::from(fixed_point_chip.dequantization({x}) as u64))))}};"]

    def _build_EqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, EqualFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = fixed_point_chip.qsub(ctx, {lhs}, {rhs});",
            f"let tmp_2 = range_chip.is_less_than(ctx, tmp_1, Constant(fixed_point_chip.quantization(0.001)), 128);",
            f"let tmp_3 = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), tmp_1, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.and(ctx, tmp_2, tmp_3);"
        ]

    def _build_NotEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, NotEqualFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = fixed_point_chip.qsub(ctx, {lhs}, {rhs});",
            f"let tmp_2 = range_chip.is_less_than(ctx, tmp_1, Constant(fixed_point_chip.quantization(-0.001)), 128);",
            f"let tmp_3 = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(0.001)), tmp_1, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.or(ctx, tmp_2, tmp_3);"
        ]

    def _build_LessThanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LessThanFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);"]

    def _build_LessThanOrEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LessThanOrEqualFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let tmp_2 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.or(ctx, tmp_1, tmp_2);"
        ]

    def _build_GreaterThanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, GreaterThanFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let tmp_2 = gate.not(ctx, tmp_1);",
            f"let tmp_3 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let tmp_4 = gate.not(ctx, tmp_3);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.and(ctx, tmp_2, tmp_4);"
        ]

    def _build_GreaterThanOrEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, GreaterThanOrEqualFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_EqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, EqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.is_equal(ctx, {lhs}, {rhs});"]

    def _build_NotEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, NotEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_LessThanIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LessThanIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);"]

    def _build_LessThanOrEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LessThanOrEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let tmp_2 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.or(ctx, tmp_1, tmp_2);"
        ]

    def _build_GreaterThanIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, GreaterThanIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let tmp_2 = gate.not(ctx, tmp_1);",
            f"let tmp_3 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let tmp_4 = gate.not(ctx, tmp_3);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.and(ctx, tmp_2, tmp_4);"
        ]

    def _build_GreaterThanOrEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, GreaterThanOrEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_BoolCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, BoolCastIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let tmp_1 = gate.is_equal(ctx, {x}, Constant(F::ZERO)));",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_LogicalNotIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LogicalNotIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, {x});"
        ]

    def _build_LogicalAndIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LogicalAndIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.and(ctx, {lhs}, {rhs});"
        ]

    def _build_LogicalOrIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LogicalOrIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.or(ctx, {lhs}, {rhs});"
        ]

    def _build_SinFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, SinFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qsin(ctx, {x});"
        ]

    def _build_ExpFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ExpFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qexp(ctx, {x});"
        ]

    def _build_LogFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LogFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qlog(ctx, {x});"
        ]

    def _build_CosFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, CosFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qcos(ctx, {x});"
        ]

    def _build_TanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, TanFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qtan(ctx, {x});"
        ]

    def _build_SinHFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, SinHFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qsinh(ctx, {x});"
        ]

    def _build_CosHFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, CosHFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qcosh(ctx, {x});"
        ]

    def _build_TanHIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, TanHFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qtanh(ctx, {x});"
        ]

    def _build_PowFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, PowFIR)
        x = self._get_var_name(stmt.arguments[0])
        exponent = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qpow(ctx, {x}, {exponent});"
        ]

    def _build_PowIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, PowIIR)
        x = self._get_var_name(stmt.arguments[0])
        exponent = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.pow_var(ctx, {x}, {exponent}, 128);"
        ]

    def _build_ModFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ModFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qmod(ctx, {lhs}, {rhs});"
        ]

    def _build_ModIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ModIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let (tmp_1, tmp_2) = range_chip.div_mod_var(ctx, {lhs}, {rhs}, 128, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = tmp_2;"
        ]

    def _build_FloorDivFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, FloorDivFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = fixed_point_chip.qmod(ctx, {lhs}, {rhs});",
            f"let tmp_2 = fixed_point_chip.qsub(ctx, {lhs}, tmp_1);",
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qdiv(ctx, tmp_2, {rhs});"
        ]

    def _build_FloorDivIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, FloorDivIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let (tmp_1, tmp_2) = range_chip.div_mod_var(ctx, {lhs}, {rhs}, 128, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = tmp_1;"
        ]

    def _build_SignFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, SignFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.sign(ctx, {x});"
        ]

    def _build_SignIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, SignIIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {x}, Constant(F::from(0)), 128);"
            f"let tmp_2 = gate.is_equal(ctx, {x}, Constant(F::from(0)), 128);"
            f"let tmp_3 = gate.select(ctx, Constant(F::from(0)), Constant(F::from(1)), tmp_2);"
            f"let tmp_4 = gate.neg(ctx, Constant(F::from(1)));"
            f"let {self._get_var_name(stmt.stmt_id)} = gate.select(ctx, tmp_4, tmp_3, tmp_1);"
        ]

    def _build_AbsFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, AbsFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qabs(ctx, {x});"
        ]

    def _build_AbsIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, AbsIIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {x}, Constant(F::from(0)), 128);"
            f"let tmp_2 = gate.neg(ctx, {x});"
            f"let {self._get_var_name(stmt.stmt_id)} = gate.select(ctx, tmp_2, {x}, tmp_1);"
        ]

    def _build_SelectIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, SelectIIR)
        cond = self._get_var_name(stmt.arguments[0])
        true_val = self._get_var_name(stmt.arguments[1])
        false_val = self._get_var_name(stmt.arguments[2])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.select(ctx, {true_val}, {false_val}, {cond});"
        ]

    def _build_SelectFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, SelectFIR)
        cond = self._get_var_name(stmt.arguments[0])
        true_val = self._get_var_name(stmt.arguments[1])
        false_val = self._get_var_name(stmt.arguments[2])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.select(ctx, {true_val}, {false_val}, {cond});"
        ]


class Halo2ProgramBuilder(AbstractProgramBuilder):
    def __init__(self, stmts: List[IRStatement], prog_metadata: ProgramMetadata):
        super().__init__(stmts, prog_metadata)

    def build(self) -> Halo2ZKProgram:
        source = self.build_source()
        return Halo2ZKProgram(self.prog_metadata.circuit_name, source)

    def build_source(self) -> str:
        return self.build_imports() + "\n" + self.build_input_data_structure() + "\n" + self.build_circuit_fn() + "\n" + self.build_main_func() + "\n"

    def build_imports(self) -> str:
        return """\
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::{Constant, Existing, Witness},
};
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;
"""

    def build_input_data_structure(self) -> str:
        inputs = []
        for i, input_obj in enumerate(self.prog_metadata.inputs):
            if isinstance(input_obj.dt, IntegerDTDescriptor):
                inputs.append(f"pub x_{i}_0: i128")
            elif isinstance(input_obj.dt, FloatDTDescriptor):
                inputs.append(f"pub x_{i}_0: f64")
            elif isinstance(input_obj.dt, NDArrayDTDescriptor):
                elements_amount = input_obj.dt.get_number_of_elements()
                for j in range(elements_amount):
                    if isinstance(input_obj.dt.dtype, IntegerDTDescriptor):
                        inputs.append(f"pub x_{i}_{j}: i128")
                    elif isinstance(input_obj.dt.dtype, FloatDTDescriptor):
                        inputs.append(f"pub x_{i}_{j}: f64")
                    else:
                        raise NotImplementedError("Unsupported NDArray dtype")
            else:
                raise NotImplementedError("Unsupported circuit input datatype")
        return """\
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
""" + ",\n".join(inputs) + "\n}\n"

    def build_circuit_fn(self) -> str:
        circuit_name = self.prog_metadata.circuit_name
        func_header = f"""\
fn {circuit_name}<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where  F: BigPrimeField {{
"""
        func_body = self.build_circuit_body()
        return func_header + func_body + "\n}"

    def build_main_func(self) -> str:
        circuit_name = self.prog_metadata.circuit_name
        return f"""\
fn main() {{
    env_logger::init();
    let args = Cli::parse();
    run({circuit_name}, args);
}}"""

    def build_circuit_body(self) -> str:
        internal_builder = _Halo2StatementBuilder()
        translated_stmts = []
        initialize_stmts = """\
    const PRECISION: u32 = 63;
    println!("build_lookup_bit: {:?}", builder.lookup_bits());
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
"""
        for stmt in self.stmts:
            translated_stmts += internal_builder.build_stmt(stmt)
        return initialize_stmts + "    " + "\n    ".join(translated_stmts)
