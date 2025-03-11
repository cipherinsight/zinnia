from typing import List

from zinnia.compile.backend.abstract_builder import AbstractProgramBuilder
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.ir_def.defs.ir_abs_f import AbsFIR
from zinnia.ir_def.defs.ir_abs_i import AbsIIR
from zinnia.ir_def.defs.ir_add_f import AddFIR
from zinnia.ir_def.defs.ir_add_i import AddIIR
from zinnia.ir_def.defs.ir_add_str import AddStrIR
from zinnia.ir_def.defs.ir_assert import AssertIR
from zinnia.ir_def.defs.ir_bool_cast import BoolCastIR
from zinnia.ir_def.defs.ir_constant_bool import ConstantBoolIR
from zinnia.ir_def.defs.ir_constant_int import ConstantIntIR
from zinnia.ir_def.defs.ir_constant_float import ConstantFloatIR
from zinnia.ir_def.defs.ir_constant_str import ConstantStrIR
from zinnia.ir_def.defs.ir_cos_f import CosFIR
from zinnia.ir_def.defs.ir_cosh_f import CosHFIR
from zinnia.ir_def.defs.ir_div_f import DivFIR
from zinnia.ir_def.defs.ir_div_i import DivIIR
from zinnia.ir_def.defs.ir_eq_f import EqualFIR
from zinnia.ir_def.defs.ir_eq_hash import EqualHashIR
from zinnia.ir_def.defs.ir_eq_i import EqualIIR
from zinnia.ir_def.defs.ir_exp_f import ExpFIR
from zinnia.ir_def.defs.ir_expose_public_f import ExposePublicFIR
from zinnia.ir_def.defs.ir_expose_public_i import ExposePublicIIR
from zinnia.ir_def.defs.ir_float_cast import FloatCastIR
from zinnia.ir_def.defs.ir_floor_divide_f import FloorDivFIR
from zinnia.ir_def.defs.ir_floor_divide_i import FloorDivIIR
from zinnia.ir_def.defs.ir_gt_f import GreaterThanFIR
from zinnia.ir_def.defs.ir_gt_i import GreaterThanIIR
from zinnia.ir_def.defs.ir_gte_f import GreaterThanOrEqualFIR
from zinnia.ir_def.defs.ir_gte_i import GreaterThanOrEqualIIR
from zinnia.ir_def.defs.ir_poseidon_hash import PoseidonHashIR
from zinnia.ir_def.defs.ir_int_cast import IntCastIR
from zinnia.ir_def.defs.ir_log_f import LogFIR
from zinnia.ir_def.defs.ir_logical_or import LogicalOrIR
from zinnia.ir_def.defs.ir_lt_f import LessThanFIR
from zinnia.ir_def.defs.ir_lt_i import LessThanIIR
from zinnia.ir_def.defs.ir_lte_f import LessThanOrEqualFIR
from zinnia.ir_def.defs.ir_lte_i import LessThanOrEqualIIR
from zinnia.ir_def.defs.ir_mod_f import ModFIR
from zinnia.ir_def.defs.ir_mod_i import ModIIR
from zinnia.ir_def.defs.ir_mul_f import MulFIR
from zinnia.ir_def.defs.ir_mul_i import MulIIR
from zinnia.ir_def.defs.ir_ne_f import NotEqualFIR
from zinnia.ir_def.defs.ir_ne_i import NotEqualIIR
from zinnia.ir_def.defs.ir_logical_not import LogicalNotIR
from zinnia.ir_def.defs.ir_logical_and import LogicalAndIR
from zinnia.ir_def.defs.ir_pow_f import PowFIR
from zinnia.ir_def.defs.ir_pow_i import PowIIR
from zinnia.ir_def.defs.ir_print import PrintIR
from zinnia.ir_def.defs.ir_read_float import ReadFloatIR
from zinnia.ir_def.defs.ir_read_hash import ReadHashIR
from zinnia.ir_def.defs.ir_read_integer import ReadIntegerIR
from zinnia.ir_def.defs.ir_select_b import SelectBIR
from zinnia.ir_def.defs.ir_select_f import SelectFIR
from zinnia.ir_def.defs.ir_select_i import SelectIIR
from zinnia.ir_def.defs.ir_sign_f import SignFIR
from zinnia.ir_def.defs.ir_sign_i import SignIIR
from zinnia.ir_def.defs.ir_sin_f import SinFIR
from zinnia.ir_def.defs.ir_sinh_f import SinHFIR
from zinnia.ir_def.defs.ir_sqrt_f import SqrtFIR
from zinnia.ir_def.defs.ir_str_f import StrFIR
from zinnia.ir_def.defs.ir_str_i import StrIIR
from zinnia.ir_def.defs.ir_sub_f import SubFIR
from zinnia.ir_def.defs.ir_sub_i import SubIIR
from zinnia.ir_def.defs.ir_tan_f import TanFIR
from zinnia.ir_def.defs.ir_tanh_f import TanHFIR


class _Halo2StatementBuilder:
    def __init__(self):
        self.id_var_lookup = {}
        self.id_val_lookup = {}

    def build_stmt(self, stmt: IRStatement) -> str:
        typename = type(stmt.ir_instance).__name__
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

    def _get_w_var_name(self, _id: int) -> str:
        return f"w_{_id}"

    def _build_AddFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AddFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qadd(ctx, {lhs}, {rhs});"]

    def _build_SubFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SubFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qsub(ctx, {lhs}, {rhs});"]

    def _build_MulFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, MulFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qmul(ctx, {lhs}, {rhs});"]

    def _build_DivFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, DivFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qdiv(ctx, {lhs}, {rhs});"]

    def _build_AddIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AddIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.add(ctx, {lhs}, {rhs});"]

    def _build_SubIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SubIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.sub(ctx, {lhs}, {rhs});"]

    def _build_MulIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, MulIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.mul(ctx, {lhs}, {rhs});"]

    def _build_DivIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, DivIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.div_unsafe(ctx, {lhs}, {rhs});"]

    def _build_AssertIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AssertIR)
        test = self._get_var_name(stmt.arguments[0])
        return [f"gate.assert_is_const(ctx, &{test}, &F::ONE);"]

    def _build_ReadIntegerIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ReadIntegerIR)
        return [
            f"let tmp_1 = ctx.load_witness(F::from_u128((input.x_{'_'.join(map(str, stmt.ir_instance.indices))}).abs() as u128));",
            f"let {self._get_var_name(stmt.stmt_id)} = if input.x_{'_'.join(map(str, stmt.ir_instance.indices))} >= 0 {{tmp_1}} else {{gate.neg(ctx, tmp_1)}};"
        ]

    def _build_ReadHashIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ReadHashIR)
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = ctx.load_witness(F::from_str_vartime(&input.hash_{'_'.join(map(str, stmt.ir_instance.indices))}).expect(\"deserialize field element should not fail\"));"
        ]

    def _build_ReadFloatIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ReadFloatIR)
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = ctx.load_witness(fixed_point_chip.quantization(input.x_{'_'.join(map(str, stmt.ir_instance.indices))}));"
        ]

    def _build_PoseidonHashIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PoseidonHashIR)
        args = [self._get_var_name(arg) for arg in stmt.arguments]
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = poseidon_hasher.hash_fix_len_array(ctx, &gate, &[{', '.join(args)}]);"
        ]

    def _build_ExposePublicIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ExposePublicIIR)
        x = self._get_var_name(stmt.arguments[0])
        return [f"make_public.push({x});"]

    def _build_ExposePublicFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ExposePublicFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [f"make_public.push({x});"]

    def _build_ConstantIntIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ConstantIntIR)
        constant_val = stmt.ir_instance.value
        if abs(constant_val) >= 18446744073709551615:
            return [
                f"let tmp_1 = ctx.load_constant(F::from_u128(\"{constant_val}\".parse::<u128>().unwrap()));",
                f"let {self._get_var_name(stmt.stmt_id)} = " + (f"tmp_1;" if constant_val >= 0 else f"{{gate.neg(ctx, tmp_1)}};")
            ]
        else:
            return [
                f"let {self._get_var_name(stmt.stmt_id)} = " + (f"Constant(F::from({constant_val}));" if constant_val >= 0 else f"{{gate.neg(ctx, Constant(F::from({-constant_val})))}};")
            ]

    def _build_ConstantBoolIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ConstantBoolIR)
        constant_val = stmt.ir_instance.value
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = Constant(F::{'ONE' if constant_val else 'ZERO'});"
        ]

    def _build_ConstantFloatIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ConstantFloatIR)
        constant_val = stmt.ir_instance.value
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = Constant(fixed_point_chip.quantization({constant_val} as f64));"
        ]

    def _build_ConstantStrIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ConstantStrIR)
        return []

    def _build_FloatCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, FloatCastIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let tmp_1 = {x}.value().get_lower_128();",
            f"let tmp_2 = gate.neg(ctx, {x});"
            f"let tmp_3 = tmp_2.value().get_lower_128();",
            f"let tmp_4 = range_chip.is_less_than(ctx, {x}, Constant(F::from(0)), 128);",
            f"let tmp_5 = tmp_4.value().get_lower_128() != 0;",
            f"let tmp_6 = if tmp_5 {{ctx.load_witness(fixed_point_chip.quantization(-(tmp_3 as f64)))}} else {{ctx.load_witness(fixed_point_chip.quantization(tmp_1 as f64))}};",
            f"let {self._get_var_name(stmt.stmt_id)} = tmp_6;"
        ]

    def _build_IntCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, IntCastIR)
        x = self._get_var_name(stmt.arguments[0])
        return [f"let {self._get_var_name(stmt.stmt_id)} = if fixed_point_chip.dequantization({x}) >= 0 {{Constant(F::from(fixed_point_chip.dequantization({x}) as u64))}} else {{gate.neg(ctx, Constant(F::from(fixed_point_chip.dequantization({x}) as u64))))}};"]

    def _build_StrIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, StrIIR)
        return []

    def _build_StrFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, StrFIR)
        return []

    def _build_PrintIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PrintIR)
        return []

    def _build_EqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, EqualFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = fixed_point_chip.qsub(ctx, {lhs}, {rhs});",
            f"let tmp_2 = range_chip.is_less_than(ctx, tmp_1, Constant(fixed_point_chip.quantization(0.001)), 128);",
            f"let tmp_3 = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), tmp_1, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.and(ctx, tmp_2, tmp_3);"
        ]

    def _build_NotEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, NotEqualFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = fixed_point_chip.qsub(ctx, {lhs}, {rhs});",
            f"let tmp_2 = range_chip.is_less_than(ctx, tmp_1, Constant(fixed_point_chip.quantization(-0.001)), 128);",
            f"let tmp_3 = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(0.001)), tmp_1, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.or(ctx, tmp_2, tmp_3);"
        ]

    def _build_LessThanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);"]

    def _build_LessThanOrEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanOrEqualFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {rhs}, {lhs}, 128);",  # Note the order of lhs and rhs
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_GreaterThanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = range_chip.is_less_than(ctx, {rhs}, {lhs}, 128);"  # Note the order of lhs and rhs
        ]

    def _build_GreaterThanOrEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanOrEqualFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_EqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, EqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.is_equal(ctx, {lhs}, {rhs});"]

    def _build_EqualHashIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, EqualHashIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.is_equal(ctx, {lhs}, {rhs});"
        ]

    def _build_NotEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, NotEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_LessThanIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [f"let {self._get_var_name(stmt.stmt_id)} = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);"]

    def _build_LessThanOrEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanOrEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {rhs}, {lhs}, 128);",  # Note the order of lhs and rhs
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_GreaterThanIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = range_chip.is_less_than(ctx, {rhs}, {lhs}, 128);"  # Note the order of lhs and rhs
        ]

    def _build_GreaterThanOrEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanOrEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_BoolCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, BoolCastIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let tmp_1 = gate.is_equal(ctx, {x}, Constant(F::ZERO)));",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_LogicalNotIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogicalNotIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, {x});"
        ]

    def _build_LogicalAndIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogicalAndIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.and(ctx, {lhs}, {rhs});"
        ]

    def _build_LogicalOrIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogicalOrIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.or(ctx, {lhs}, {rhs});"
        ]

    def _build_SinFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SinFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qsin(ctx, {x});"
        ]

    def _build_ExpFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ExpFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qexp(ctx, {x});"
        ]

    def _build_LogFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qlog(ctx, {x});"
        ]

    def _build_CosFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, CosFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qcos(ctx, {x});"
        ]

    def _build_TanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, TanFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qtan(ctx, {x});"
        ]

    def _build_SinHFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SinHFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qsinh(ctx, {x});"
        ]

    def _build_CosHFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, CosHFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qcosh(ctx, {x});"
        ]

    def _build_TanHIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, TanHFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qtanh(ctx, {x});"
        ]

    def _build_PowFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PowFIR)
        x = self._get_var_name(stmt.arguments[0])
        exponent = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qpow(ctx, {x}, {exponent});"
        ]

    def _build_PowIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PowIIR)
        x = self._get_var_name(stmt.arguments[0])
        exponent = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.pow_var(ctx, {x}, {exponent}, 128);"
        ]

    def _build_ModFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ModFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qmod(ctx, {lhs}, {rhs});"
        ]

    def _build_ModIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ModIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let (tmp_1, tmp_2) = range_chip.div_mod_var(ctx, {lhs}, {rhs}, 128, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = tmp_2;"
        ]

    def _build_FloorDivFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, FloorDivFIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let tmp_1 = fixed_point_chip.qmod(ctx, {lhs}, {rhs});",
            f"let tmp_2 = fixed_point_chip.qsub(ctx, {lhs}, tmp_1);",
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qdiv(ctx, tmp_2, {rhs});"
        ]

    def _build_FloorDivIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, FloorDivIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        return [
            f"let (tmp_1, tmp_2) = range_chip.div_mod_var(ctx, {lhs}, {rhs}, 128, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = tmp_1;"
        ]

    def _build_SignFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SignFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.sign(ctx, {x});"
        ]

    def _build_SignIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SignIIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {x}, Constant(F::from(0)), 128);"
            f"let tmp_2 = gate.is_equal(ctx, {x}, Constant(F::from(0)), 128);"
            f"let tmp_3 = gate.select(ctx, Constant(F::from(0)), Constant(F::from(1)), tmp_2);"
            f"let tmp_4 = gate.neg(ctx, Constant(F::from(1)));"
            f"let {self._get_var_name(stmt.stmt_id)} = gate.select(ctx, tmp_4, tmp_3, tmp_1);"
        ]

    def _build_AbsFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AbsFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qabs(ctx, {x});"
        ]

    def _build_SqrtFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SqrtFIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qsqrt(ctx, {x});"
        ]

    def _build_AbsIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AbsIIR)
        x = self._get_var_name(stmt.arguments[0])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {x}, Constant(F::from(0)), 128);"
            f"let tmp_2 = gate.neg(ctx, {x});"
            f"let {self._get_var_name(stmt.stmt_id)} = gate.select(ctx, tmp_2, {x}, tmp_1);"
        ]

    def _build_SelectIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SelectIIR)
        cond = self._get_var_name(stmt.arguments[0])
        true_val = self._get_var_name(stmt.arguments[1])
        false_val = self._get_var_name(stmt.arguments[2])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.select(ctx, {true_val}, {false_val}, {cond});"
        ]

    def _build_SelectBIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SelectBIR)
        cond = self._get_var_name(stmt.arguments[0])
        true_val = self._get_var_name(stmt.arguments[1])
        false_val = self._get_var_name(stmt.arguments[2])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.select(ctx, {true_val}, {false_val}, {cond});"
        ]

    def _build_SelectFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SelectFIR)
        cond = self._get_var_name(stmt.arguments[0])
        true_val = self._get_var_name(stmt.arguments[1])
        false_val = self._get_var_name(stmt.arguments[2])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.select(ctx, {true_val}, {false_val}, {cond});"
        ]

    def _build_AddStrIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AddStrIR)
        return []


class Halo2ProgramBuilder(AbstractProgramBuilder):
    input_entries: List

    def __init__(self, name: str, stmts: List[IRStatement]):
        super().__init__(name, stmts)
        self.input_entries = []
        for stmt in stmts:
            op = stmt.ir_instance
            if isinstance(op, ReadIntegerIR):
                self.input_entries.append((op.indices, "Integer"))
            elif isinstance(op, ReadFloatIR):
                self.input_entries.append((op.indices, "Float"))
            elif isinstance(op, ReadHashIR):
                self.input_entries.append((op.indices, "Hash"))

    def build(self) -> str:
        return self.build_source()

    def build_source(self) -> str:
        return self.build_imports() + "\n" + self.build_input_data_structure() + "\n" + self.build_circuit_fn() + "\n" + self.build_main_func() + "\n"

    def build_imports(self) -> str:
        return """\
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
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
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;
"""

    def build_input_data_structure(self) -> str:
        inputs = []
        for i, (indices, kind) in enumerate(self.input_entries):
            if kind == "Hash":
                inputs.append(f"pub hash_{'_'.join(map(str, indices))}: String")
            elif kind == "Integer":
                inputs.append(f"pub x_{'_'.join(map(str, indices))}: i128")
            elif kind == "Float":
                inputs.append(f"pub x_{'_'.join(map(str, indices))}: f64")
            else:
                raise NotImplementedError("Unsupported circuit input datatype")
        return """\
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
""" + ",\n".join(inputs) + "\n}\n"

    def build_circuit_fn(self) -> str:
        circuit_name = self.name
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
        circuit_name = self.name
        return f"""\
fn main() {{
    env_logger::init();
    let args = Cli::parse();
    run({circuit_name}, args);
}}"""

    def build_circuit_body(self) -> str:
        internal_builder = _Halo2StatementBuilder()
        translated_stmts = []
        has_poseidon_hash_ir = any(isinstance(stmt.ir_instance, PoseidonHashIR) for stmt in self.stmts)
        initialize_stmts = """\
    const PRECISION: u32 = 63;
    println!("build_lookup_bit: {:?}", builder.lookup_bits());
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let mut poseidon_hasher = PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);
"""
        if has_poseidon_hash_ir:
            initialize_stmts += "    poseidon_hasher.initialize_consts(ctx, &gate);\n"
        for stmt in self.stmts:
            translated_stmts += internal_builder.build_stmt(stmt)
        return initialize_stmts + "    " + "\n    ".join(translated_stmts)
