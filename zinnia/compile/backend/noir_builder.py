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
from zinnia.ir_def.defs.ir_inv_i import InvIIR
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


class _NoirStatementBuilder:
    def __init__(self):
        self.id_var_lookup = {}
        self.next_tmp_id = 0
        self.id_val_lookup = {}
        self.id_type_lookup = {}

    def build_stmt(self, stmt: IRStatement) -> str:
        typename = type(stmt.ir_instance).__name__
        method_name = '_build_' + typename
        method = getattr(self, method_name, None)
        if method is None:
            raise NotImplementedError(method_name)
        return method(stmt)

    def _get_var_name(self, _id: int, _type: str) -> str:
        var_name = self.id_var_lookup.get(_id, None)
        if var_name is not None:
            if _type != self.id_type_lookup[_id]:
                if _type == "field" and self.id_type_lookup[_id] == "bool":
                    return f"(if {var_name} {{1}} else {{0}})"
                elif _type == "bool" and self.id_type_lookup[_id] == "field":
                    return f"({var_name} != 0)"
            return var_name
        var_name = f"y_{_id}"
        self.id_var_lookup[_id] = var_name
        self.id_type_lookup[_id] = _type
        return var_name

    def _build_AddFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AddFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_SubFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SubFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_MulFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, MulFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_DivFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, DivFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_FloorDivFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, FloorDivFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_AddIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AddIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name} = {lhs} + {rhs};",
        ]

    def _build_SubIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SubIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name} = {lhs} - {rhs};",
        ]

    def _build_MulIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, MulIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name} = {lhs} * {rhs};",
        ]

    def _build_DivIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, DivIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name} = ({lhs} as Field) / ({rhs} as Field);",
        ]

    def _build_FloorDivIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, FloorDivIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name} = (({lhs} as u128) / ({rhs} as u128)) as Field;"
        ]

    def _build_AssertIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AssertIR)
        test = self._get_var_name(stmt.arguments[0], "bool")
        return [
            f"assert({test});"
        ]

    def _build_ReadIntegerIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ReadIntegerIR)
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name}: Field = x_{'_'.join(map(str, stmt.ir_instance.indices))};"
        ]

    def _build_ReadHashIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ReadHashIR)
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name}: Field = x_{'_'.join(map(str, stmt.ir_instance.indices))};"
        ]

    def _build_ReadFloatIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ReadFloatIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_PoseidonHashIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PoseidonHashIR)
        assign_inputs = [
            f"{self._get_var_name(arg, 'field')}"
            for i, arg in enumerate(stmt.arguments)
        ]
        text = ', '.join(assign_inputs)
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let inputs: [Field; {len(stmt.arguments)}] = [{text}];",
            f"let {var_name}: Field = poseidon::bn254::hash_10(inputs);"
        ]

    def _build_ExposePublicIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ExposePublicIIR)
        return []

    def _build_ExposePublicFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ExposePublicFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_ConstantIntIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ConstantIntIR)
        constant_val = stmt.ir_instance.value
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name} = {int(constant_val)};"
        ]

    def _build_ConstantBoolIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ConstantBoolIR)
        constant_val = stmt.ir_instance.value
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = {'true' if constant_val else 'false'};"
        ]

    def _build_ConstantFloatIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ConstantFloatIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_ConstantStrIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ConstantStrIR)
        return []

    def _build_FloatCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, FloatCastIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_IntCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, IntCastIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_StrIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, StrIIR)
        return []

    def _build_StrFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, StrFIR)
        return []

    def _build_PrintIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PrintIR)
        return []

    def _build_SinFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SinFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_ExpFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ExpFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_LogFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_CosFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, CosFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_TanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, TanFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_SinHFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SinHFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_CosHFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, CosHFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_TanHIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, TanHFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_PowFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PowFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_PowIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PowIIR)
        x = self._get_var_name(stmt.arguments[0], "field")
        exponent = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let mut {var_name}: Field = 1;",
            f"for i in 0..252 {{",
            f"    {var_name} = {var_name} * {var_name};",
            f"    let mask: u128 = ((({exponent} as u128) >> ((251 - (i as u8)) as u8))) & 1;",
            f"    if mask == 1 {{",
            f"        {var_name} = {var_name} * {x};",
            f"    }}",
            f"}}"
        ]

    def _build_LogicalNotIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogicalNotIR)
        x = self._get_var_name(stmt.arguments[0], "bool")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = !({x});"
        ]

    def _build_LogicalAndIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogicalAndIR)
        lhs = self._get_var_name(stmt.arguments[0], "bool")
        rhs = self._get_var_name(stmt.arguments[1], "bool")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = ({lhs} & {rhs});"
        ]

    def _build_LogicalOrIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogicalOrIR)
        lhs = self._get_var_name(stmt.arguments[0], "bool")
        rhs = self._get_var_name(stmt.arguments[1], "bool")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = ({lhs} | {rhs});"
        ]

    def _build_BoolCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, BoolCastIR)
        x = self._get_var_name(stmt.arguments[0], "field")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = if {x} == 0 {{false}} else {{true}};",
        ]

    def _build_EqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, EqualFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_NotEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, NotEqualFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_LessThanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_LessThanOrEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanOrEqualFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_GreaterThanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_GreaterThanOrEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanOrEqualFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_EqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, EqualIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = {lhs} == {rhs};"
        ]

    def _build_EqualHashIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, EqualHashIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = {lhs} == {rhs};"
        ]

    def _build_NotEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, NotEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = {lhs} != {rhs};"
        ]

    def _build_LessThanIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = ({lhs} as u128) < ({rhs} as u128);"
        ]

    def _build_LessThanOrEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanOrEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = ({lhs} as u128) <= ({rhs} as u128);"
        ]

    def _build_GreaterThanIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = ({lhs} as u128) > ({rhs} as u128);"
        ]

    def _build_GreaterThanOrEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanOrEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = ({lhs} as u128) >= ({rhs} as u128);"
        ]

    def _build_AbsFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AbsFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_SqrtFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SqrtFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_AbsIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AbsIIR)
        x = self._get_var_name(stmt.arguments[0], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name}: Field = if ({x} as u128) > 10944121435919637611123202872628637544274182200208017171849102093287904247808 {{0 - {x}}} else {x};",
        ]

    def _build_SignFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SignFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_SignIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SignIIR)
        x = self._get_var_name(stmt.arguments[0], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name}: Field = if ({x} as u128) > 10944121435919637611123202872628637544274182200208017171849102093287904247808 {{1}} else {{0 - 1}};",
        ]

    def _build_SelectIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SelectIIR)
        cond = self._get_var_name(stmt.arguments[0], "bool")
        true_val = self._get_var_name(stmt.arguments[1], "field")
        false_val = self._get_var_name(stmt.arguments[2], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name} = if {cond} {{ {true_val} }} else {{ {false_val} }};"
        ]

    def _build_SelectBIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SelectBIR)
        cond = self._get_var_name(stmt.arguments[0], "bool")
        true_val = self._get_var_name(stmt.arguments[1], "bool")
        false_val = self._get_var_name(stmt.arguments[2], "bool")
        var_name = self._get_var_name(stmt.stmt_id, "bool")
        return [
            f"let {var_name} = if {cond} {{ {true_val} }} else {{ {false_val} }};"
        ]

    def _build_SelectFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SelectFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_AddStrIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AddStrIR)
        return []

    def _build_ModFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ModFIR)
        raise NotImplementedError("Floating-point operations are not supported in Noir")

    def _build_ModIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ModIIR)
        lhs = self._get_var_name(stmt.arguments[0], "field")
        rhs = self._get_var_name(stmt.arguments[1], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name} = (({lhs} as u128) % ({rhs} as u128)) as Field;",
        ]

    def _build_InvIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, InvIIR)
        x = self._get_var_name(stmt.arguments[0], "field")
        var_name = self._get_var_name(stmt.stmt_id, "field")
        return [
            f"let {var_name}: Field = 1 / ({x} as Field);"
        ]


class NoirProgramBuilder(AbstractProgramBuilder):
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
        imports_str = self.build_imports()
        circuit_fn_str = self.build_circuit_template()
        return imports_str  + "\n" + circuit_fn_str + "\n"

    def build_imports(self) -> str:
        return "use std::hash::poseidon;\n"

    def build_circuit_template(self) -> str:
        func_header = f"""\
fn main({self.build_main_params()}) {{
"""
        func_body = self.build_circuit_body()
        return func_header + "    " + func_body + "\n}"

    def build_main_params(self) -> str:
        the_inputs = []
        for stmt in self.stmts:
            if isinstance(stmt.ir_instance, ReadIntegerIR):
                if stmt.ir_instance.is_public:
                    the_inputs.append(f'x_{"_".join(map(str, stmt.ir_instance.indices))}: pub Field')
                else:
                    the_inputs.append(f'x_{"_".join(map(str, stmt.ir_instance.indices))}: Field')
            elif isinstance(stmt.ir_instance, ReadFloatIR):
                raise NotImplementedError("Floating-point operations are not supported in Noir")
            elif isinstance(stmt.ir_instance, ReadHashIR):
                the_inputs.append(f"x_{'_'.join(map(str, stmt.ir_instance.indices))}: pub Field")
        return ', '.join(the_inputs)

    def build_circuit_body(self) -> str:
        internal_builder = _NoirStatementBuilder()
        translated_stmts = []
        for stmt in self.stmts:
            translated_stmts += internal_builder.build_stmt(stmt)
        return "\n    ".join(translated_stmts)
