from typing import List
from zinnia.debug.dbg_info import DebugInfo
import re

from zinnia.compile.backend.abstract_builder import AbstractProgramBuilder
from zinnia.compile.ir.ir_stmt import IRStatement
from zinnia.ir_def.defs.ir_abs_f import AbsFIR
from zinnia.ir_def.defs.ir_abs_i import AbsIIR
from zinnia.ir_def.defs.ir_add_f import AddFIR
from zinnia.ir_def.defs.ir_add_i import AddIIR
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


# thought it could be fun to see what piecewise translation would look like via dictionary lookup
# will the power of a python dict lead to speedy transpiling?
# we will see...

class CarrieHalo2ProgramBuilder(AbstractProgramBuilder):
    ir_to_rust_conversions: dict

    def __init__(self, name: str, stmts: List[IRStatement]):
        super().__init__(name, stmts)
        self.build_ir_to_rust_conversions()


    # TODO: rest of IR statements for this one
    def build_ir_to_rust_conversions(self):
        self.ir_to_rust_conversions = {
            AddFIR: ("let y_%s = fixed_point_chip.qadd(ctx, %s, %s);", ("id", "arg_0", "arg_1")),
            SubFIR: ("let y_%s = fixed_point_chip.qsub(ctx, %s, %s);", ("id", "arg_0", "arg_1")),
            MulFIR: ("let y_%s = fixed_point_chip.qmul(ctx, %s, %s);", ("id", "arg_0", "arg_1")),
            DivFIR: ("let y_%s = fixed_point_chip.qdiv(ctx, %s, %s);", ("id", "arg_0", "arg_1")),
            AddIIR: ("let y_%s = gate.add(ctx, %s, %s);", ("id", "arg_0", "arg_1")),
            SubIIR: ("let y_%s = gate.sub(ctx, %s, %s);", ("id", "arg_0", "arg_1")),
            MulIIR: ("let y_%s = gate.mul(ctx, %s, %s);", ("id", "arg_0", "arg_1")),
            DivIIR: ("let y_%s = gate.div_unsafe(ctx, %s, %s);", ("id", "arg_0", "arg_1")),
            AssertIR: ("gate.assert_is_const(ctx, &%s, &F::ONE);", ("arg_0",)),
            ReadIntegerIR: ("let tmp_1 = ctx.load_witness(F::from_u128((input.x_%s).abs() as u128));\nlet y_%s = if input.x_%s >= 0 {{tmp_1}} else {{gate.neg(ctx, tmp_1)}};", ("indices", "id", "indices")),
            ReadHashIR: ("let y_%s = ctx.load_witness(F::from_str_vartime(&input.hash_%s).expect(\"deserialize field element should not fail\"));", ("id", "indices")),
            ReadFloatIR: ("let y_%s = ctx.load_witness(fixed_point_chip.quantization(input.x_%s));", ("id", "indices")),
            PoseidonHashIR: ("let y_%s = poseidon_hasher.hash_fix_len_array(ctx, &gate, &[%s]);", ("id", "comma_args")),
            ExposePublicIIR: ("make_public.push(%s);", ("arg_0",)),
            ExposePublicFIR: ("make_public.push(%s);", ("arg_0",)),
            ConstantIntIR: ("let y_%s = ctx.load_constant(F::from_u128(%s as u128));", ("id", "value")),
            ConstantFloatIR: ("let y_%s = Constant(fixed_point_chip.quantization(%s as f64));", ("id", "value")),
            EqualIIR: ("let y_%s = gate.is_equal(ctx, y_%s, y_%s);", ("id", "arg_0", "arg_1")),
            NotEqualIIR: ("let tmp_1 = gate.is_equal(ctx, %s, %s);\nlet y_%s = gate.not(ctx, tmp_1);", ("arg_0", "arg_1", "id"))
        }


    # TODO: fix source code formatting issue in terminal
    def build(self) -> str:
        return self.build_source()


    def build_source(self) -> str:
        program_body = self.build_inputs() + "\n" + self.build_circuit_func() + "\n" + self.build_main_func() + "\n"
        imports = self.build_imports(program_body)
        return imports + "\n" + program_body


    # create circuit input struct
    def build_inputs(self) -> str:
        rust_input_vars: list[str] = []

        # iterate through ir statements to find 'read' statements
        for stmt in self.stmts:
            stmt_instance_type = stmt.ir_instance
            match stmt_instance_type:
                case ReadIntegerIR():
                    rust_input_vars.append('pub x_%s: i128' % "_".join(map(str, stmt_instance_type.indices)))
                case ReadFloatIR():
                    rust_input_vars.append('pub x_%s: f64' % "_".join(map(str, stmt_instance_type.indices)))
                case ReadHashIR():
                    rust_input_vars.append('pub x_%s: String' % "_".join(map(str, stmt_instance_type.indices)))
        
        circuit_input_code = """
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {\n""" + ",\n".join(rust_input_vars) + "\n}\n"
        return circuit_input_code



    def build_circuit_func(self) -> str:
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
    
    
    def build_circuit_body(self):
        initialize_stmts = """\
    const PRECISION: u32 = 63;
    println!("build_lookup_bit: {:?}", builder.lookup_bits());
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let mut poseidon_hasher = PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);
"""

        translated_stmts: list[str] = []
        has_poseidon_hash_ir = any(isinstance(stmt.ir_instance, PoseidonHashIR) for stmt in self.stmts)
        if has_poseidon_hash_ir:
            initialize_stmts += "    poseidon_hasher.initialize_consts(ctx, &gate);\n"
        for stmt in self.stmts:
            translated_stmts += self.build_stmt(stmt)

        translated_stmts_str = "\n\t".join(translated_stmts)
        return initialize_stmts + translated_stmts_str
    
    # arg types:
    # id (stmt id)
    # indices (joined with _)
    # arg_0 (arg's id)
    # arg_1 (arg's id)
    # comma_args (do later)
    # value (ir instance.value)

    def build_stmt(self, stmt: IRStatement) -> str:
        # do dict lookup here
        try:
            code_args_tuple: tuple[str, tuple[str]] = self.ir_to_rust_conversions[stmt.ir_instance.__class__]
        except KeyError:
            raise NotImplementedError(f"Internal Error: IR class {stmt.ir_instance.__class__} not located in builder dictionary.")
        
        # extract code + required arguments to fill in here
        arg_values: list[str] = []
        arg_names: tuple[str] = code_args_tuple[1]

        for arg in arg_names:
            match arg:
                case "id": arg_values.append(stmt.stmt_id)
                case "indices": arg_values.append("_".join(map(str, stmt.ir_instance.indices)))
                case "arg_0": arg_values.append(stmt.arguments[0])
                case "arg_1": arg_values.append(stmt.arguments[1])
                case "value": arg_values.append(stmt.ir_instance.value)
                case _: raise NotImplementedError("Internal Error: argument type '%s' has not yet been implemented in build_stmt" % arg)
        
        rust_translation: str = (code_args_tuple[0] % tuple(arg_values)) + "\n\t"
        return rust_translation


    

    # this is just the most straightforward way to do it tbh
    def build_main_func(self) -> str:
        circuit_name = self.name
        return f"""\
fn main() {{
    env_logger::init();
    let args = Cli::parse();
    run({circuit_name}, args);
}}"""

    # check imports needed based on string matching once the rest of the circuit has been made
    def build_imports(self, program_body: str) -> str:
        import_stmts = ""
        import_translations: dict[str, str] = {
            "ScalarField": "use halo2_base::utils::{ScalarField};",
            "BigPrimeField": "use halo2_base::utils::{BigPrimeField};",
            "FixedPointChip": "use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};",
            "BaseCircuitBuilder": "use halo2_base::gates::circuit::builder::BaseCircuitBuilder;",
            "PoseidonHasher": "use halo2_base::poseidon::hasher::PoseidonHasher;",
            "GateChip": "use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};",
            "Serialize": "use serde::{Serialize, Deserialize};",
            "AssignedValue<[^>]+>": "use halo2_base::{Context, AssignedValue};",
            "Existing(AssignedValue<[^>]+>)": "use halo2_base::QuantumCell::Existing;",
            "Constant\([^>]+\)": "use halo2_base::QuantumCell::Constant;",
            "Witness\([^>]+\)": "use halo2_base::QuantumCell::Witness;",
            "WitnessFraction\(Assigned<[^>]+>\)": "use halo2_base::QuantumCell::WitnessFraction;",
            "Cli::": "use halo2_graph::scaffold::cmd::Cli;",
            "run\([^)]*,[^)]*\)": "use halo2_graph::scaffold::run;",
            "OptimizedPoseidonSpec": "use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;"
        }

        for key in import_translations.keys():
            if re.search(key, program_body):
                import_stmts += (import_translations[key] + "\n")

        import_stmts += """\
const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;
"""
        
        return import_stmts