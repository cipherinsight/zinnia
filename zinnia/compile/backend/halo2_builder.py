from typing import List
import json

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


class _Opcode:
    ADD_F=1; SUB_F=2; MUL_F=3; DIV_F=4
    ADD_I=5; SUB_I=6; MUL_I=7; DIV_I=8
    CONST_I=9; CONST_F=10; CONST_B=11
    READ_I=12; READ_F=13; READ_HASH=14
    EQ_F=15; NE_F=16; LT_F=17; LE_F=18; GT_F=19; GE_F=20
    EQ_I=21; NE_I=22; LT_I=23; LE_I=24; GT_I=25; GE_I=26
    BOOL_NOT=27; BOOL_AND=28; BOOL_OR=29; BOOL_CAST=30
    SELECT_I=31; SELECT_F=32; SELECT_B=33
    MOD_I=34; MOD_F=35; FLOORDIV_I=36; FLOORDIV_F=37
    POW_I=38; POW_F=39
    ABS_I=40; ABS_F=41; SQRT_F=42; SIGN_I=43; SIGN_F=44
    SIN_F=45; COS_F=46; TAN_F=47; SINH_F=48; COSH_F=49; TANH_F=50
    INT_CAST=51; FLOAT_CAST=52
    ASSERT=60; EXPOSE_I=61; EXPOSE_F=62; POSEIDON=63

class _InstrEncoder:
    """
    Encodes IR into compact tuples AND records input bindings so we can preload
    regs for Read* without generating huge Rust per-instruction code.
    """
    def __init__(self, input_entries):
        # input_entries: list of ((indices), "Integer"/"Float"/"Hash")
        # Build a map: ("Integer", "x_1_2") etc. for fast lookup
        self.input_kinds = {}  # field_name -> kind
        for indices, kind in input_entries:
            if kind == "Hash":
                name = f"hash_{'_'.join(map(str, indices))}"
            else:
                name = f"x_{'_'.join(map(str, indices))}"
            self.input_kinds[name] = kind

        self.max_reg = 0
        self.read_bindings = []  # {dst, kind, field}

    def _dst(self, stmt):
        self.max_reg = max(self.max_reg, stmt.stmt_id)
        return stmt.stmt_id

    def _arg(self, ridx):
        self.max_reg = max(self.max_reg, ridx)
        return ridx

    def _field_from_indices(self, kind, indices):
        if kind == "Hash":
            return f"hash_{'_'.join(map(str, indices))}"
        else:
            return f"x_{'_'.join(map(str, indices))}"

    def encode(self, stmt: IRStatement):
        op = stmt.ir_instance
        sid = self._dst(stmt)
        args = stmt.arguments

        def a(i): return self._arg(args[i]) if len(args) > i else 0

        # ---- arithmetic (float) ----
        from zinnia.ir_def.defs.ir_add_f import AddFIR
        from zinnia.ir_def.defs.ir_sub_f import SubFIR
        from zinnia.ir_def.defs.ir_mul_f import MulFIR
        from zinnia.ir_def.defs.ir_div_f import DivFIR
        if isinstance(op, AddFIR): return (_Opcode.ADD_F, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, SubFIR): return (_Opcode.SUB_F, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, MulFIR): return (_Opcode.MUL_F, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, DivFIR): return (_Opcode.DIV_F, sid, a(0), a(1), 0, 0, 0.0)

        # ---- arithmetic (int) ----
        from zinnia.ir_def.defs.ir_add_i import AddIIR
        from zinnia.ir_def.defs.ir_sub_i import SubIIR
        from zinnia.ir_def.defs.ir_mul_i import MulIIR
        from zinnia.ir_def.defs.ir_div_i import DivIIR
        if isinstance(op, AddIIR): return (_Opcode.ADD_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, SubIIR): return (_Opcode.SUB_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, MulIIR): return (_Opcode.MUL_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, DivIIR): return (_Opcode.DIV_I, sid, a(0), a(1), 0, 0, 0.0)

        # ---- constants ----
        from zinnia.ir_def.defs.ir_constant_int import ConstantIntIR
        from zinnia.ir_def.defs.ir_constant_float import ConstantFloatIR
        from zinnia.ir_def.defs.ir_constant_bool import ConstantBoolIR
        if isinstance(op, ConstantIntIR): return (_Opcode.CONST_I, sid, 0, 0, 0, int(op.value), 0.0)
        if isinstance(op, ConstantFloatIR): return (_Opcode.CONST_F, sid, 0, 0, 0, 0, float(op.value))
        if isinstance(op, ConstantBoolIR): return (_Opcode.CONST_B, sid, 0, 0, 0, 1 if op.value else 0, 0.0)

        # ---- IO (record binding; do not emit heavy code) ----
        from zinnia.ir_def.defs.ir_read_integer import ReadIntegerIR
        from zinnia.ir_def.defs.ir_read_float import ReadFloatIR
        from zinnia.ir_def.defs.ir_read_hash import ReadHashIR
        if isinstance(op, ReadIntegerIR):
            field = self._field_from_indices("Integer", op.indices)
            self.read_bindings.append({"dst": sid, "kind": "Integer", "field": field})
            return None
        if isinstance(op, ReadFloatIR):
            field = self._field_from_indices("Float", op.indices)
            self.read_bindings.append({"dst": sid, "kind": "Float", "field": field})
            return None
        if isinstance(op, ReadHashIR):
            field = self._field_from_indices("Hash", op.indices)
            self.read_bindings.append({"dst": sid, "kind": "Hash", "field": field})
            return None

        # ---- comparisons, logical, select, math, casts, expose/assert/hash ----
        # (imports condensed for brevity; identical to your earlier mapping)
        from zinnia.ir_def.defs.ir_eq_f import EqualFIR
        from zinnia.ir_def.defs.ir_ne_f import NotEqualFIR
        from zinnia.ir_def.defs.ir_lt_f import LessThanFIR
        from zinnia.ir_def.defs.ir_lte_f import LessThanOrEqualFIR
        from zinnia.ir_def.defs.ir_gt_f import GreaterThanFIR
        from zinnia.ir_def.defs.ir_gte_f import GreaterThanOrEqualFIR
        if isinstance(op, EqualFIR): return (_Opcode.EQ_F, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, NotEqualFIR): return (_Opcode.NE_F, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, LessThanFIR): return (_Opcode.LT_F, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, LessThanOrEqualFIR): return (_Opcode.LE_F, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, GreaterThanFIR): return (_Opcode.GT_F, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, GreaterThanOrEqualFIR): return (_Opcode.GE_F, sid, a(0), a(1), 0, 0, 0.0)

        from zinnia.ir_def.defs.ir_eq_i import EqualIIR
        from zinnia.ir_def.defs.ir_ne_i import NotEqualIIR
        from zinnia.ir_def.defs.ir_lt_i import LessThanIIR
        from zinnia.ir_def.defs.ir_lte_i import LessThanOrEqualIIR
        from zinnia.ir_def.defs.ir_gt_i import GreaterThanIIR
        from zinnia.ir_def.defs.ir_gte_i import GreaterThanOrEqualIIR
        if isinstance(op, EqualIIR): return (_Opcode.EQ_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, NotEqualIIR): return (_Opcode.NE_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, LessThanIIR): return (_Opcode.LT_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, LessThanOrEqualIIR): return (_Opcode.LE_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, GreaterThanIIR): return (_Opcode.GT_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, GreaterThanOrEqualIIR): return (_Opcode.GE_I, sid, a(0), a(1), 0, 0, 0.0)

        from zinnia.ir_def.defs.ir_logical_not import LogicalNotIR
        from zinnia.ir_def.defs.ir_logical_and import LogicalAndIR
        from zinnia.ir_def.defs.ir_logical_or import LogicalOrIR
        from zinnia.ir_def.defs.ir_bool_cast import BoolCastIR
        if isinstance(op, LogicalNotIR): return (_Opcode.BOOL_NOT, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, LogicalAndIR): return (_Opcode.BOOL_AND, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, LogicalOrIR):  return (_Opcode.BOOL_OR,  sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, BoolCastIR):   return (_Opcode.BOOL_CAST, sid, a(0), 0, 0, 0, 0.0)

        from zinnia.ir_def.defs.ir_select_i import SelectIIR
        from zinnia.ir_def.defs.ir_select_f import SelectFIR
        from zinnia.ir_def.defs.ir_select_b import SelectBIR
        if isinstance(op, SelectIIR): return (_Opcode.SELECT_I, sid, a(0), a(1), a(2), 0, 0.0)
        if isinstance(op, SelectFIR): return (_Opcode.SELECT_F, sid, a(0), a(1), a(2), 0, 0.0)
        if isinstance(op, SelectBIR): return (_Opcode.SELECT_B, sid, a(0), a(1), a(2), 0, 0.0)

        from zinnia.ir_def.defs.ir_mod_i import ModIIR
        from zinnia.ir_def.defs.ir_mod_f import ModFIR
        from zinnia.ir_def.defs.ir_floor_divide_i import FloorDivIIR
        from zinnia.ir_def.defs.ir_floor_divide_f import FloorDivFIR
        from zinnia.ir_def.defs.ir_pow_i import PowIIR
        from zinnia.ir_def.defs.ir_pow_f import PowFIR
        if isinstance(op, ModIIR):      return (_Opcode.MOD_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, ModFIR):      return (_Opcode.MOD_F, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, FloorDivIIR): return (_Opcode.FLOORDIV_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, FloorDivFIR): return (_Opcode.FLOORDIV_F, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, PowIIR):      return (_Opcode.POW_I, sid, a(0), a(1), 0, 0, 0.0)
        if isinstance(op, PowFIR):      return (_Opcode.POW_F, sid, a(0), a(1), 0, 0, 0.0)

        from zinnia.ir_def.defs.ir_abs_i import AbsIIR
        from zinnia.ir_def.defs.ir_abs_f import AbsFIR
        from zinnia.ir_def.defs.ir_sqrt_f import SqrtFIR
        from zinnia.ir_def.defs.ir_sign_i import SignIIR
        from zinnia.ir_def.defs.ir_sign_f import SignFIR
        from zinnia.ir_def.defs.ir_sin_f import SinFIR
        from zinnia.ir_def.defs.ir_cos_f import CosFIR
        from zinnia.ir_def.defs.ir_tan_f import TanFIR
        from zinnia.ir_def.defs.ir_sinh_f import SinHFIR
        from zinnia.ir_def.defs.ir_cosh_f import CosHFIR
        from zinnia.ir_def.defs.ir_tanh_f import TanHFIR
        if isinstance(op, AbsIIR):  return (_Opcode.ABS_I, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, AbsFIR):  return (_Opcode.ABS_F, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, SqrtFIR): return (_Opcode.SQRT_F, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, SignIIR): return (_Opcode.SIGN_I, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, SignFIR): return (_Opcode.SIGN_F, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, SinFIR):  return (_Opcode.SIN_F, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, CosFIR):  return (_Opcode.COS_F, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, TanFIR):  return (_Opcode.TAN_F, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, SinHFIR): return (_Opcode.SINH_F, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, CosHFIR): return (_Opcode.COSH_F, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, TanHFIR): return (_Opcode.TANH_F, sid, a(0), 0, 0, 0, 0.0)

        from zinnia.ir_def.defs.ir_int_cast import IntCastIR
        from zinnia.ir_def.defs.ir_float_cast import FloatCastIR
        if isinstance(op, IntCastIR):   return (_Opcode.INT_CAST, sid, a(0), 0, 0, 0, 0.0)
        if isinstance(op, FloatCastIR): return (_Opcode.FLOAT_CAST, sid, a(0), 0, 0, 0, 0.0)

        from zinnia.ir_def.defs.ir_assert import AssertIR
        from zinnia.ir_def.defs.ir_expose_public_i import ExposePublicIIR
        from zinnia.ir_def.defs.ir_expose_public_f import ExposePublicFIR
        from zinnia.ir_def.defs.ir_poseidon_hash import PoseidonHashIR
        if isinstance(op, AssertIR):       return (_Opcode.ASSERT, 0, a(0), 0, 0, 0, 0.0)
        if isinstance(op, ExposePublicIIR):return (_Opcode.EXPOSE_I, 0, a(0), 0, 0, 0, 0.0)
        if isinstance(op, ExposePublicFIR):return (_Opcode.EXPOSE_F, 0, a(0), 0, 0, 0, 0.0)
        if isinstance(op, PoseidonHashIR): return (_Opcode.POSEIDON, sid, a(0), a(1), a(2) if len(args)>2 else 0, 0, 0.0)

        return None

def _emit_meta_comment(program, reads, num_regs):
    meta = {
        "version": 1,
        "num_regs": num_regs,
        "program": [
            {"op": op, "dst": dst, "a": a, "b": b, "c": c, "imm_i": imm_i, "imm_f": imm_f}
            for (op, dst, a, b, c, imm_i, imm_f) in program
        ],
        "reads": reads
    }
    payload = json.dumps(meta, separators=(',',':'))
    return f"""/*<BEGIN_ZINNIA_META>
{payload}
</END_ZINNIA_META>*/"""


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
        # Prepare encoded program + meta
        enc = _InstrEncoder(self.input_entries)
        program = []
        for s in self.stmts:
            item = enc.encode(s)
            if item is not None:
                program.append(item)
        meta_comment = _emit_meta_comment(program, enc.read_bindings, enc.max_reg + 1)

        # Splice into source
        return (
            meta_comment + "\n" +               # <-- JSON-in-comment header
            self.build_imports() + "\n" +
            self.build_input_data_structure() + "\n" +
            self.build_circuit_fn(program, enc) + "\n" +
            self.build_main_func() + "\n"
        )

    def build_main_func(self) -> str:
        circuit_name = self.name
        return f"""\
fn main() {{
    env_logger::init();
    let args = Cli::try_parse().unwrap();
    run({circuit_name}, args);
}}"""

    def build_imports(self) -> str:
        return """\
    use std::fs;
    use clap::Parser;
    use halo2_base::utils::{ScalarField, BigPrimeField};
    use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
    use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
    use halo2_base::poseidon::hasher::PoseidonHasher;
    use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
    use serde::{Serialize, Deserialize};
    use serde_json;
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
                inputs.append(f"pub x_{'_'.join(map(str, indices))}: String")
            elif kind == "Float":
                inputs.append(f"pub x_{'_'.join(map(str, indices))}: f64")
            else:
                raise NotImplementedError("Unsupported circuit input datatype")
        return """\
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
""" + ",\n".join(inputs) + "\n}\n"

    def build_circuit_fn(self, program, enc) -> str:
        circuit_name = self.name
        func_header = f"""\
fn {circuit_name}<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where  F: BigPrimeField {{
"""
        func_body = self.build_circuit_body(program, enc)
        return func_header + func_body + "\n}"


    def build_circuit_body(self, program, enc) -> str:
        has_poseidon = any(isinstance(s.ir_instance, PoseidonHashIR) for s in self.stmts)

        # helper: generate small matchers to fetch input fields by name
        # (keeps the runtime generic while avoiding reflection)
        int_arms  = []
        float_arms= []
        hash_arms = []
        for (indices, kind) in self.input_entries:
            if kind == "Hash":
                field = f"hash_{'_'.join(map(str, indices))}"
                hash_arms.append(f"\"{field}\" => Some(&input.{field}),")
            elif kind == "Integer":
                field = f"x_{'_'.join(map(str, indices))}"
                int_arms.append(f"\"{field}\" => Some(&input.{field}),")
            elif kind == "Float":
                field = f"x_{'_'.join(map(str, indices))}"
                float_arms.append(f"\"{field}\" => Some(input.{field}),")
        int_arms_s   = "\n            ".join(int_arms)   or "_ => None,"
        float_arms_s = "\n            ".join(float_arms) or "_ => None,"
        hash_arms_s  = "\n            ".join(hash_arms)  or "_ => None,"

        # rust: declare Instr / ReadBinding / Meta and the dispatcher (same opcodes)
        rust = f"""\
        const PRECISION: u32 = 63;
        println!("build_lookup_bit: {{:?}}", builder.lookup_bits());
        let gate = GateChip::<F>::default();
        let range_chip = builder.range_chip();
        let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
        let mut poseidon_hasher = PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
        let ctx = builder.main(0);
    """ + ("    poseidon_hasher.initialize_consts(ctx, &gate);\n" if has_poseidon else "") + f"""
        // ===== Self-describing meta: read from our own source file =====
        // Replace PLACHOLDER_PATH at build time with the absolute path to THIS .rs file.
        let src = fs::read_to_string("/home/zhantong/halo2-graph/examples/target.rs").expect("read source");
        let start_tag = "<BEGIN_ZINNIA_META>";
        let end_tag   = "</END_ZINNIA_META>";
        let start = src.find(start_tag).expect("begin tag not found") + start_tag.len();
        let end   = src.find(end_tag).expect("end tag not found");
        let json_blob = &src[start..end];

        #[derive(Clone, Copy, Deserialize)]
        struct Instr {{ op:u8, dst:u32, a:u32, b:u32, c:u32, imm_i:i128, imm_f:f64 }}
        #[derive(Clone, Deserialize)]
        struct ReadBinding {{ dst:u32, kind:String, field:String }}
        #[derive(Clone, Deserialize)]
        struct Meta {{ version:u32, num_regs:u32, program:Vec<Instr>, reads:Vec<ReadBinding> }}

        // Opcodes (must match the Python encoder)
        const ADD_F:u8=1;const SUB_F:u8=2;const MUL_F:u8=3;const DIV_F:u8=4;
        const ADD_I:u8=5;const SUB_I:u8=6;const MUL_I:u8=7;const DIV_I:u8=8;
        const CONST_I:u8=9;const CONST_F:u8=10;const CONST_B:u8=11;
        const READ_I:u8=12;const READ_F:u8=13;const READ_HASH:u8=14;
        const EQ_F:u8=15;const NE_F:u8=16;const LT_F:u8=17;const LE_F:u8=18;const GT_F:u8=19;const GE_F:u8=20;
        const EQ_I:u8=21;const NE_I:u8=22;const LT_I:u8=23;const LE_I:u8=24;const GT_I:u8=25;const GE_I:u8=26;
        const BOOL_NOT:u8=27;const BOOL_AND:u8=28;const BOOL_OR:u8=29;const BOOL_CAST:u8=30;
        const SELECT_I:u8=31;const SELECT_F:u8=32;const SELECT_B:u8=33;
        const MOD_I:u8=34;const MOD_F:u8=35;const FLOORDIV_I:u8=36;const FLOORDIV_F:u8=37;
        const POW_I:u8=38;const POW_F:u8=39;
        const ABS_I:u8=40;const ABS_F:u8=41;const SQRT_F:u8=42;const SIGN_I:u8=43;const SIGN_F:u8=44;
        const SIN_F:u8=45;const COS_F:u8=46;const TAN_F:u8=47;const SINH_F:u8=48;const COSH_F:u8=49;const TANH_F:u8=50;
        const INT_CAST:u8=51;const FLOAT_CAST:u8=52;
        const ASSERT:u8=60;const EXPOSE_I:u8=61;const EXPOSE_F:u8=62;const POSEIDON:u8=63;

        let meta: Meta = serde_json::from_str(json_blob).expect("parse meta json");

        // Register file
        let mut regs: Vec<AssignedValue<F>> = vec![ctx.load_constant(F::ZERO); meta.num_regs as usize];

        // ---- helper closures to fetch inputs by name ----
        let get_int = |name:&str| -> Option<&String> {{
            match name {{
                {int_arms_s}
                _ => None,
            }}
        }};
        let get_f64 = |name:&str| -> Option<f64> {{
            match name {{
                {float_arms_s}
                _ => None,
            }}
        }};
        let get_hash = |name:&str| -> Option<&String> {{
            match name {{
                {hash_arms_s}
                _ => None,
            }}
        }};

        // ---- preload READ_* from side-table ----
        for rb in &meta.reads {{
            if rb.kind == "Integer" {{
                let s = get_int(&rb.field).expect("int field missing");
                let neg = s.chars().next() == Some('-');
                let slice = if neg {{ &s[1..] }} else {{ &s[..] }};
                let val = F::from_str_vartime(slice).expect("int parse");
                let w = ctx.load_witness(val);
                regs[rb.dst as usize] = if neg {{ gate.neg(ctx, w) }} else {{ w }};
            }} else if rb.kind == "Float" {{
                let v = get_f64(&rb.field).expect("float field missing");
                regs[rb.dst as usize] = ctx.load_witness(fixed_point_chip.quantization(v));
            }} else if rb.kind == "Hash" {{
                let s = get_hash(&rb.field).expect("hash field missing");
                let val = F::from_str_vartime(s).expect("hash parse");
                regs[rb.dst as usize] = ctx.load_witness(val);
            }} else {{
                panic!("unknown read kind");
            }}
        }}

        // ---- execute program ----
        for ins in meta.program {{
            match ins.op {{
                // constants
                CONST_I => {{
                    let v = ins.imm_i;
                    let av = if v >= 0 {{ ctx.load_constant(F::from_u128(v as u128)) }}
                             else {{ gate.neg(ctx, Constant(F::from_u128((-v) as u128))) }};
                    regs[ins.dst as usize] = av;
                }},
                CONST_F => {{
                    let q = fixed_point_chip.quantization(ins.imm_f);
                    regs[ins.dst as usize] = ctx.load_constant(q);
                }},
                CONST_B => {{
                    regs[ins.dst as usize] = if ins.imm_i != 0 {{ ctx.load_constant(F::ONE) }} else {{ ctx.load_constant(F::ZERO) }};
                }},

                // float arith
                ADD_F => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= fixed_point_chip.qadd(ctx,a,b); }},
                SUB_F => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= fixed_point_chip.qsub(ctx,a,b); }},
                MUL_F => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= fixed_point_chip.qmul(ctx,a,b); }},
                DIV_F => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= fixed_point_chip.qdiv(ctx,a,b); }},

                // int arith
                ADD_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= gate.add(ctx,a,b); }},
                SUB_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= gate.sub(ctx,a,b); }},
                MUL_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= gate.mul(ctx,a,b); }},
                DIV_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= gate.div_unsafe(ctx,a,b); }},

                // cmp float (tolerance ±1e-3)
                EQ_F => {{
                    let a=regs[ins.a as usize]; let b=regs[ins.b as usize];
                    let d = fixed_point_chip.qsub(ctx,a,b);
                    let le = range_chip.is_less_than(ctx, d, Constant(fixed_point_chip.quantization(0.001)), 128);
                    let ge = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), d, 128);
                    regs[ins.dst as usize] = gate.and(ctx, le, ge);
                }},
                NE_F => {{
                    let a=regs[ins.a as usize]; let b=regs[ins.b as usize];
                    let d = fixed_point_chip.qsub(ctx,a,b);
                    let lt = range_chip.is_less_than(ctx, d, Constant(fixed_point_chip.quantization(-0.001)), 128);
                    let gt = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(0.001)), d, 128);
                    regs[ins.dst as usize] = gate.or(ctx, lt, gt);
                }},
                LT_F => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= range_chip.is_less_than(ctx,a,b,128); }},
                LE_F => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; let t=range_chip.is_less_than(ctx,b,a,128); regs[ins.dst as usize]= gate.not(ctx,t); }},
                GT_F => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= range_chip.is_less_than(ctx,b,a,128); }},
                GE_F => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; let t=range_chip.is_less_than(ctx,a,b,128); regs[ins.dst as usize]= gate.not(ctx,t); }},

                // cmp int
                EQ_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= gate.is_equal(ctx,a,b); }},
                NE_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; let t=gate.is_equal(ctx,a,b); regs[ins.dst as usize]= gate.not(ctx,t); }},
                LT_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= range_chip.is_less_than(ctx,a,b,128); }},
                LE_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; let t=range_chip.is_less_than(ctx,b,a,128); regs[ins.dst as usize]= gate.not(ctx,t); }},
                GT_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= range_chip.is_less_than(ctx,b,a,128); }},
                GE_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; let t=range_chip.is_less_than(ctx,a,b,128); regs[ins.dst as usize]= gate.not(ctx,t); }},

                // logic
                BOOL_NOT => {{ let x=regs[ins.a as usize]; regs[ins.dst as usize]= gate.not(ctx,x); }},
                BOOL_AND => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= gate.and(ctx,a,b); }},
                BOOL_OR  => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= gate.or(ctx,a,b); }},
                BOOL_CAST => {{
                    let x=regs[ins.a as usize];
                    let z = gate.is_equal(ctx, x, Constant(F::ZERO));
                    regs[ins.dst as usize] = gate.not(ctx, z);
                }},

                // select
                SELECT_I | SELECT_F | SELECT_B => {{
                    let cond=regs[ins.a as usize]; let tval=regs[ins.b as usize]; let fval=regs[ins.c as usize];
                    regs[ins.dst as usize] = gate.select(ctx, tval, fval, cond);
                }},

                // mod/div/pow
                MOD_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; let (_q,r)=range_chip.div_mod_var(ctx,a,b,128,128); regs[ins.dst as usize]= r; }},
                FLOORDIV_I => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; let (q,_r)=range_chip.div_mod_var(ctx,a,b,128,128); regs[ins.dst as usize]= q; }},
                MOD_F => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; regs[ins.dst as usize]= fixed_point_chip.qmod(ctx,a,b); }},
                FLOORDIV_F => {{ let a=regs[ins.a as usize]; let b=regs[ins.b as usize]; let r=fixed_point_chip.qmod(ctx,a,b); let s=fixed_point_chip.qsub(ctx,a,r); regs[ins.dst as usize]= fixed_point_chip.qdiv(ctx,s,b); }},
                POW_I => {{ let x=regs[ins.a as usize]; let e=regs[ins.b as usize]; regs[ins.dst as usize]= gate.pow_var(ctx,x,e,128); }},
                POW_F => {{ let x=regs[ins.a as usize]; let e=regs[ins.b as usize]; regs[ins.dst as usize]= fixed_point_chip.qpow(ctx,x,e); }},

                // math/unary
                ABS_I => {{ let x=regs[ins.a as usize]; let neg=gate.neg(ctx,x); let lt0=range_chip.is_less_than(ctx,x,Constant(F::from(0)),128);
                            regs[ins.dst as usize]= gate.select(ctx, neg, x, lt0); }},
                ABS_F => {{ let x=regs[ins.a as usize]; regs[ins.dst as usize]= fixed_point_chip.qabs(ctx,x); }},
                SQRT_F => {{ let x=regs[ins.a as usize]; regs[ins.dst as usize]= fixed_point_chip.qsqrt(ctx,x); }},
                SIGN_I => {{
                    let x=regs[ins.a as usize];
                    let lt=range_chip.is_less_than(ctx,x,Constant(F::from(0)),128);
                    let eq=gate.is_equal(ctx,x,Constant(F::from(0)));
                    let z1=gate.select(ctx, Constant(F::from(0)), Constant(F::from(1)), eq);
                    let neg1=gate.neg(ctx, Constant(F::from(1)));
                    regs[ins.dst as usize]= gate.select(ctx, neg1, z1, lt);
                }},
                SIGN_F => {{ let x=regs[ins.a as usize]; regs[ins.dst as usize]= fixed_point_chip.sign(ctx,x); }},
                SIN_F => {{ let x=regs[ins.a as usize]; regs[ins.dst as usize]= fixed_point_chip.qsin(ctx,x); }},
                COS_F => {{ let x=regs[ins.a as usize]; regs[ins.dst as usize]= fixed_point_chip.qcos(ctx,x); }},
                TAN_F => {{ let x=regs[ins.a as usize]; regs[ins.dst as usize]= fixed_point_chip.qtan(ctx,x); }},
                SINH_F => {{ let x=regs[ins.a as usize]; regs[ins.dst as usize]= fixed_point_chip.qsinh(ctx,x); }},
                COSH_F => {{ let x=regs[ins.a as usize]; regs[ins.dst as usize]= fixed_point_chip.qcosh(ctx,x); }},
                TANH_F => {{ let x=regs[ins.a as usize]; regs[ins.dst as usize]= fixed_point_chip.qtanh(ctx,x); }},
                INT_CAST => {{
                    let x=regs[ins.a as usize];
                    let v = fixed_point_chip.dequantization(*x.value());
                    let av = if v >= 0.0 {{ ctx.load_constant(F::from(v as u64)) }} else {{ gate.neg(ctx, Constant(F::from((-v) as u64))) }};
                    regs[ins.dst as usize] = av;
                }},
                FLOAT_CAST => {{
                    // Keep your FloatCastIR semantics approximation; adjust to your field API if needed.
                    let x=regs[ins.a as usize];
                    let lt0 = range_chip.is_less_than(ctx, x, Constant(F::from(0)), 128);
                    let is_neg = lt0.value().get_lower_128() != 0;
                    let mag = if is_neg {{ gate.neg(ctx, x) }} else {{ x }};
                    // This is a placeholder path; you might refine with a safer conversion
                    let lo = mag.value().get_lower_128() as f64;
                    let q = if is_neg {{ fixed_point_chip.quantization(-lo) }} else {{ fixed_point_chip.quantization(lo) }};
                    regs[ins.dst as usize] = ctx.load_witness(q);
                }},

                // poseidon
                POSEIDON => {{
                    let a=regs[ins.a as usize];
                    let b=regs[ins.b as usize];
                    let c= if ins.c != 0 {{ regs[ins.c as usize] }} else {{ ctx.load_constant(F::ZERO) }};
                    regs[ins.dst as usize] = poseidon_hasher.hash_fix_len_array(ctx, &gate, &[a,b,c]);
                }},

                // assert/expose
                ASSERT => {{ let t=regs[ins.a as usize]; gate.assert_is_const(ctx, &t, &F::ONE); }},
                EXPOSE_I | EXPOSE_F => {{ let x=regs[ins.a as usize]; make_public.push(x); }},

                _ => {{}}
            }}
        }}
    """
        return rust
