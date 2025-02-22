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
from zinnia.ir_def.defs.ir_select_f import SelectFIR
from zinnia.ir_def.defs.ir_select_i import SelectIIR
from zinnia.ir_def.defs.ir_sign_f import SignFIR
from zinnia.ir_def.defs.ir_sign_i import SignIIR
from zinnia.ir_def.defs.ir_sin_f import SinFIR
from zinnia.ir_def.defs.ir_sinh_f import SinHFIR
from zinnia.ir_def.defs.ir_str_f import StrFIR
from zinnia.ir_def.defs.ir_str_i import StrIIR
from zinnia.ir_def.defs.ir_sub_f import SubFIR
from zinnia.ir_def.defs.ir_sub_i import SubIIR
from zinnia.ir_def.defs.ir_tan_f import TanFIR
from zinnia.ir_def.defs.ir_tanh_f import TanHFIR


class _CircomStatementBuilder:
    def __init__(self):
        self.id_var_lookup = {}
        self.next_tmp_id = 0
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

    def _allocate_tmp_name(self) -> str:
        var_name = f"t_{self.next_tmp_id}"
        self.next_tmp_id += 1
        return var_name

    def _build_AddFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AddFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_SubFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SubFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_MulFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, MulFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_DivFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, DivFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_FloorDivFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, FloorDivFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_AddIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AddIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== {lhs} + {rhs};",
        ]

    def _build_SubIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SubIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== {lhs} - {rhs};",
        ]

    def _build_MulIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, MulIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== {lhs} * {rhs};",
        ]

    def _build_DivIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, DivIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <-- {lhs} / {rhs};",
            f"{var_name} * {rhs} === {lhs};",
        ]

    def _build_FloorDivIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, FloorDivIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <-- {lhs} / {rhs};",
            f"{var_name} * {rhs} === {lhs};",
        ]

    def _build_AssertIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AssertIR)
        test = self._get_var_name(stmt.arguments[0])
        return [
            f"{test} === 1;"
        ]

    def _build_ReadIntegerIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ReadIntegerIR)
        return [
            f"signal input x_{'_'.join(map(str, stmt.ir_instance.indices))};",
            f"signal {self._get_var_name(stmt.stmt_id)} <== x_{'_'.join(map(str, stmt.ir_instance.indices))};"
        ]

    def _build_ReadHashIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ReadHashIR)
        return [
            f"signal input hash_{'_'.join(map(str, stmt.ir_instance.indices))};",
            f"signal {self._get_var_name(stmt.stmt_id)} <== hash_{'_'.join(map(str, stmt.ir_instance.indices))};"
        ]

    def _build_ReadFloatIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ReadFloatIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_PoseidonHashIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PoseidonHashIR)
        tmp_name_1 = self._allocate_tmp_name()
        assign_inputs_statements = [
            f"{tmp_name_1}.inputs[{i}] <== {self._get_var_name(arg)};"
            for i, arg in enumerate(stmt.arguments)
        ]
        return [
            f"component {tmp_name_1} = Poseidon({len(stmt.arguments)});",
            *assign_inputs_statements,
            f"signal {self._get_var_name(stmt.stmt_id)} <== {tmp_name_1}.out;"
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
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== {constant_val};"
        ]

    def _build_ConstantFloatIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ConstantFloatIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_ConstantStrIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ConstantStrIR)
        return []

    def _build_FloatCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, FloatCastIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_IntCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, IntCastIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

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
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_ExpFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ExpFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_LogFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_CosFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, CosFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_TanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, TanFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_SinHFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SinHFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_CosHFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, CosHFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_TanHIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, TanHFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_PowFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PowFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_PowIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, PowIIR)
        x = self._get_var_name(stmt.arguments[0])
        exponent = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== ZinniaCodeGenReservedPow(252)({x}, {exponent});"
        ]

    def _build_LogicalNotIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogicalNotIR)
        x = self._get_var_name(stmt.arguments[0])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== 1 - {x};"
        ]

    def _build_LogicalAndIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogicalAndIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== {lhs} * {rhs};"
        ]

    def _build_LogicalOrIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LogicalOrIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== {lhs} + {rhs} - {lhs} * {rhs};"
        ]

    def _build_BoolCastIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, BoolCastIR)
        x = self._get_var_name(stmt.arguments[0])
        var_name = self._get_var_name(stmt.stmt_id)
        tmp_name_1 = self._allocate_tmp_name()
        return [
            f"signal {tmp_name_1} <== ZinniaCodeGenReservedIsZero()([{x}]);",
            f"signal {var_name} <== 1 - {tmp_name_1};"
        ]

    def _build_EqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, EqualFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_NotEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, NotEqualFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_LessThanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_LessThanOrEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanOrEqualFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_GreaterThanFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_GreaterThanOrEqualFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanOrEqualFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_EqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, EqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== ZinniaCodeGenReservedIsEqual()([{lhs}, {rhs}]);"
        ]

    def _build_EqualHashIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, EqualHashIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== ZinniaCodeGenReservedIsEqual()([{lhs}, {rhs}]);"
        ]

    def _build_NotEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, NotEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        tmp_name_1 = self._allocate_tmp_name()
        return [
            f"signal {tmp_name_1} <== ZinniaCodeGenReservedIsEqual()([{lhs}, {rhs}]);",
            f"signal {var_name} <== 1 - {tmp_name_1};"
        ]

    def _build_LessThanIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== ZinniaCodeGenReservedLessThan(252)([{lhs}, {rhs}]);"
        ]

    def _build_LessThanOrEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, LessThanOrEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== ZinniaCodeGenReservedLessEqThan(252)([{lhs}, {rhs}]);"
        ]

    def _build_GreaterThanIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== ZinniaCodeGenReservedGreaterThan(252)([{lhs}, {rhs}]);"
        ]

    def _build_GreaterThanOrEqualIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, GreaterThanOrEqualIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== ZinniaCodeGenReservedGreaterEqThan(252)([{lhs}, {rhs}]);"
        ]

    def _build_AbsFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AbsFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_AbsIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AbsIIR)
        x = self._get_var_name(stmt.arguments[0])
        var_name = self._get_var_name(stmt.stmt_id)
        tmp_name_1 = self._allocate_tmp_name()
        return [
            f"signal {tmp_name_1} <== ZinniaCodeGenReservedLessThan(252)([{x}, 0]);",
            f"signal {var_name} <== ZinniaCodeGenReservedSelector()([{x}, -{x}], {tmp_name_1});"
        ]

    def _build_SignFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SignFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_SignIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SignIIR)
        x = self._get_var_name(stmt.arguments[0])
        var_name = self._get_var_name(stmt.stmt_id)
        tmp_name_1 = self._allocate_tmp_name()
        tmp_name_2 = self._allocate_tmp_name()
        tmp_name_3 = self._allocate_tmp_name()
        return [
            f"signal {tmp_name_1} <== ZinniaCodeGenReservedLessThan(252)([{x, 0}]);",
            f"signal {tmp_name_2} <== ZinniaCodeGenReservedSelector()([1, -1], {tmp_name_1});",
            f"signal {tmp_name_3} <== ZinniaCodeGenReservedIsEqual()([{x, 0}]);",
            f"signal {var_name} <== ZinniaCodeGenReservedSelector()([{tmp_name_2}, 0], {tmp_name_3});"
        ]

    def _build_SelectIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SelectIIR)
        cond = self._get_var_name(stmt.arguments[0])
        true_val = self._get_var_name(stmt.arguments[1])
        false_val = self._get_var_name(stmt.arguments[2])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== ZinniaCodeGenReservedSelector()([{false_val}, {true_val}], {cond});"
        ]

    def _build_SelectFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, SelectFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_AddStrIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, AddStrIR)
        return []

    def _build_ModFIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ModFIR)
        raise NotImplementedError("Floating-point operations are not supported in Circom")

    def _build_ModIIR(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.ir_instance, ModIIR)
        lhs = self._get_var_name(stmt.arguments[0])
        rhs = self._get_var_name(stmt.arguments[1])
        var_name = self._get_var_name(stmt.stmt_id)
        return [
            f"signal {var_name} <== ZinniaCodeGenReservedIntMod(252)([{lhs}, {rhs}]);"
        ]


class CircomProgramBuilder(AbstractProgramBuilder):
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
        main_func_str = self.build_main_func()
        return imports_str  + "\n" + circuit_fn_str + "\n\n" + main_func_str

    def build_imports(self) -> str:
        return """\
pragma circom 2.1.6;

include "circomlib/poseidon.circom";

template ZinniaCodeGenReservedAssertBit() {
  signal input in;

  in * (in - 1) === 0;
}

template ZinniaCodeGenReservedNum2Bits(n) {
  assert(n < 254);
  signal input in;
  signal output out[n];

  var lc = 0;
  var bit_value = 1;

  for (var i = 0; i < n; i++) {
    out[i] <-- (in >> i) & 1;
    ZinniaCodeGenReservedAssertBit()(out[i]);

    lc += out[i] * bit_value;
    bit_value <<= 1;
  }

  lc === in;
}

template ZinniaCodeGenReservedBits2Num(n) {
  assert(n < 254);
  signal input in[n];
  signal output out;

  var lc = 0;
  var bit_value = 1;
  for (var i = 0; i < n; i++) {
    ZinniaCodeGenReservedAssertBit()(in[i]);

    lc += in[i] * bit_value;
    bit_value <<= 1;
  }

  out <== lc;
}

template ZinniaCodeGenReservedIsZero() {
  signal input in;
  signal output out;

  signal inv <-- in != 0 ? 1 / in : 0;
  out <== 1 - (in * inv);

  in * out === 0;
}

template ZinniaCodeGenReservedIsEqual() {
  signal input in[2];
  signal output out;

  out <== ZinniaCodeGenReservedIsZero()(in[1] - in[0]);
}

template ZinniaCodeGenReservedLessThan(n) {
  assert(n <= 252);
  signal input in[2];
  signal output out;

  component toBits = ZinniaCodeGenReservedNum2Bits(n+1);
  toBits.in <== ((1 << n) + in[0]) - in[1];

  out <== 1 - toBits.out[n];
}

template ZinniaCodeGenReservedLessEqThan(n) {
  signal input in[2];
  signal output out;

  out <== ZinniaCodeGenReservedLessThan(n)([in[0], in[1]+1]);
}

template ZinniaCodeGenReservedGreaterThan(n) {
  signal input in[2];
  signal output out;

  out <== ZinniaCodeGenReservedLessThan(n)([in[1], in[0]]);
}

template ZinniaCodeGenReservedGreaterEqThan(n) {
  signal input in[2];
  signal output out;

  out <== ZinniaCodeGenReservedLessThan(n)([in[1], in[0]+1]);
}

template ZinniaCodeGenReservedSelector() {
  signal input in[2];
  signal input sel;
  signal output out;

  out <== (in[1] - in[0]) * sel + in[0];
}

template ZinniaCodeGenReservedPow(nBits) {
    signal input base;
    signal input exponent;
    signal output out;

    // Decompose the exponent into its binary representation (LSB first).
    component bits = ZinniaCodeGenReservedNum2Bits(nBits);
    bits.in <== exponent;

    // We use two arrays:
    // acc: accumulates the result,
    // power: holds successive squares of the base.
    signal acc[nBits+1];
    signal tmp[nBits+1];
    signal power[nBits+1];

    // Initialize: acc[0] = 1 and power[0] = base.
    acc[0] <== 1;
    power[0] <== base;

    // For each bit, update the accumulator and square the current power.
    for (var i = 0; i < nBits; i++) {
        // If bits.out[i] == 1, then multiply acc[i] by power[i],
        // else leave acc[i] unchanged.
        // The expression (bits.out[i]*(power[i]-1) + 1) equals:
        //   - 1 when bits.out[i] is 0 (since 0*(power[i]-1)+1 = 1),
        //   - power[i] when bits.out[i] is 1 (since 1*(power[i]-1)+1 = power[i]).
        tmp[i+1] <== bits.out[i]*(power[i] - 1) + 1;
        acc[i+1] <== acc[i] * tmp[i+1];
        // Square the current power for the next iteration.
        power[i+1] <== power[i] * power[i];
    }

    // The final accumulated value is our result.
    out <== acc[nBits];
}

template ZinniaCodeGenReservedIntDiv(n) {
  signal input in[2];
  signal output out;

  // divisor must be non-zero
  signal is_non_zero <== ZinniaCodeGenReservedIsZero()(in[1]);
  0 === is_non_zero;

  // compute the quotient and remainder outside the circuit
  var quot_hint = in[0] \\ in[1];
  var rem_hint = in[0] % in[1];
  signal quot <-- quot_hint;
  signal rem <-- rem_hint;

  // contrain the division operation
  // in[0] / in[1] is defined as the unique pair (q, r) s.t.
  // in[0] = in[1] * q + r and 0 <= r < |in[1]|
  in[0] === quot * in[1] + rem;

  // OPTIONAL: quot edge case is when `rem = 0` and `in[1] = 1`
  // signal quot_is_valid <== ZinniaCodeGenReservedLessEqThan(n)([quot, in[0]]);
  // 1 === quot_is_valid;

  signal rem_is_valid <== ZinniaCodeGenReservedLessThan(n)([rem, in[1]]);
  1 === rem_is_valid;

  out <== quot;
}

template ZinniaCodeGenReservedIntMod(n) {
  signal input in[2];
  signal output out;

  // divisor must be non-zero
  signal is_non_zero <== ZinniaCodeGenReservedIsZero()(in[1]);
  0 === is_non_zero;

  // compute the quotient and remainder outside the circuit
  var quot_hint = in[0] \\ in[1];
  var rem_hint = in[0] % in[1];
  signal quot <-- quot_hint;
  signal rem <-- rem_hint;

  // contrain the division operation
  // in[0] / in[1] is defined as the unique pair (q, r) s.t.
  // in[0] = in[1] * q + r and 0 <= r < |in[1]|
  in[0] === quot * in[1] + rem;

  // OPTIONAL: quot edge case is when `rem = 0` and `in[1] = 1`
  // signal quot_is_valid <== ZinniaCodeGenReservedLessEqThan(n)([quot, in[0]]);
  // 1 === quot_is_valid;

  signal rem_is_valid <== ZinniaCodeGenReservedLessThan(n)([rem, in[1]]);
  1 === rem_is_valid;

  out <== rem;
}
"""

    def build_circuit_template(self) -> str:
        circuit_name = self.name
        func_header = f"""\
template {circuit_name}() {{
"""
        func_body = self.build_circuit_body()
        return func_header + "    " + func_body + "\n}"

    def build_main_func(self) -> str:
        circuit_name = self.name
        public_inputs = []
        for stmt in self.stmts:
            if isinstance(stmt.ir_instance, ReadIntegerIR) and stmt.ir_instance.is_public:
                public_inputs.append(f"x_{'_'.join(map(str, stmt.ir_instance.indices))}")
            elif isinstance(stmt.ir_instance, ReadFloatIR) and stmt.ir_instance.is_public:
                public_inputs.append(f"x_{'_'.join(map(str, stmt.ir_instance.indices))}")
            elif isinstance(stmt.ir_instance, ReadHashIR) and stmt.ir_instance.is_public:
                public_inputs.append(f"x_{'_'.join(map(str, stmt.ir_instance.indices))}")
        public_inputs_str = ''
        if len(public_inputs) > 0:
            public_inputs_str = f"{{ public [ {', '.join(public_inputs)} ] }}"
        return f"""\
component main {public_inputs_str} = {circuit_name}();"""

    def build_circuit_body(self) -> str:
        internal_builder = _CircomStatementBuilder()
        translated_stmts = []
        for stmt in self.stmts:
            translated_stmts += internal_builder.build_stmt(stmt)
        return "\n    ".join(translated_stmts)
