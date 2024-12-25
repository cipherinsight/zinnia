from typing import List

from pyzk.backend.abstract_builder import AbstractProgramBuilder
from pyzk.backend.zk_program import Halo2ZKProgram
from pyzk.internal.dt_descriptor import IntegerDTDescriptor, FloatDTDescriptor, NDArrayDTDescriptor
from pyzk.internal.prog_meta_data import ProgramMetadata
from pyzk.ir.ir_stmt import IRStatement
from pyzk.opdef.nocls.op_add_f import AddFOp
from pyzk.opdef.nocls.op_add_i import AddIOp
from pyzk.opdef.nocls.op_and import AndOp
from pyzk.opdef.nocls.op_assert import AssertOp
from pyzk.opdef.nocls.op_bool_cast import BoolCastOp
from pyzk.opdef.nocls.op_constant import ConstantOp
from pyzk.opdef.nocls.op_constant_float import ConstantFloatOp
from pyzk.opdef.nocls.op_div_f import DivFOp
from pyzk.opdef.nocls.op_div_i import DivIOp
from pyzk.opdef.nocls.op_eq_f import EqualFOp
from pyzk.opdef.nocls.op_eq_i import EqualIOp
from pyzk.opdef.nocls.op_float import FloatOp
from pyzk.opdef.nocls.op_gt_f import GreaterThanFOp
from pyzk.opdef.nocls.op_gt_i import GreaterThanIOp
from pyzk.opdef.nocls.op_gte_f import GreaterThanOrEqualFOp
from pyzk.opdef.nocls.op_gte_i import GreaterThanOrEqualIOp
from pyzk.opdef.nocls.op_int import IntOp
from pyzk.opdef.nocls.op_lt_f import LessThanFOp
from pyzk.opdef.nocls.op_lt_i import LessThanIOp
from pyzk.opdef.nocls.op_lte_f import LessThanOrEqualFOp
from pyzk.opdef.nocls.op_lte_i import LessThanOrEqualIOp
from pyzk.opdef.nocls.op_mul_f import MulFOp
from pyzk.opdef.nocls.op_mul_i import MulIOp
from pyzk.opdef.nocls.op_ne_f import NotEqualFOp
from pyzk.opdef.nocls.op_ne_i import NotEqualIOp
from pyzk.opdef.nocls.op_not import NotOp
from pyzk.opdef.nocls.op_or import OrOp
from pyzk.opdef.nocls.op_read_float import ReadFloatOp
from pyzk.opdef.nocls.op_read_integer import ReadIntegerOp
from pyzk.opdef.nocls.op_sub_f import SubFOp
from pyzk.opdef.nocls.op_sub_i import SubIOp


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

    def _build_AddFOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, AddFOp)
        lhs = self._get_var_name(stmt.arguments["lhs"])
        rhs = self._get_var_name(stmt.arguments["rhs"])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qadd(ctx, {lhs}, {rhs});"]

    def _build_SubFOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, SubFOp)
        lhs = self._get_var_name(stmt.arguments["lhs"])
        rhs = self._get_var_name(stmt.arguments["rhs"])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qsub(ctx, {lhs}, {rhs});"]

    def _build_MulFOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, MulFOp)
        lhs = self._get_var_name(stmt.arguments["lhs"])
        rhs = self._get_var_name(stmt.arguments["rhs"])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qmul(ctx, {lhs}, {rhs});"]

    def _build_DivFOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, DivFOp)
        lhs = self._get_var_name(stmt.arguments["lhs"])
        rhs = self._get_var_name(stmt.arguments["rhs"])
        return [f"let {self._get_var_name(stmt.stmt_id)} = fixed_point_chip.qdiv(ctx, {lhs}, {rhs});"]

    def _build_AddIOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, AddIOp)
        lhs = self._get_var_name(stmt.arguments["lhs"])
        rhs = self._get_var_name(stmt.arguments["rhs"])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.add(ctx, {lhs}, {rhs});"]

    def _build_SubIOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, SubIOp)
        lhs = self._get_var_name(stmt.arguments["lhs"])
        rhs = self._get_var_name(stmt.arguments["rhs"])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.sub(ctx, {lhs}, {rhs});"]

    def _build_MulIOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, MulIOp)
        lhs = self._get_var_name(stmt.arguments["lhs"])
        rhs = self._get_var_name(stmt.arguments["rhs"])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.mul(ctx, {lhs}, {rhs});"]

    def _build_DivIOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, DivIOp)
        lhs = self._get_var_name(stmt.arguments["lhs"])
        rhs = self._get_var_name(stmt.arguments["rhs"])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.div_unsafe(ctx, {lhs}, {rhs});"]

    def _build_AssertOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, AssertOp)
        test = self._get_var_name(stmt.arguments["test"])
        return [f"gate.assert_is_const(ctx, &{test}, &F::ONE);"]

    def _build_ReadIntegerOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ReadIntegerOp)
        major: int = stmt.operator.major
        minor: int = stmt.operator.minor
        return [
            f"let tmp_1 = ctx.load_witness(F::from((input.x_{major}_{minor}).abs() as u64));",
            f"let {self._get_var_name(stmt.stmt_id)} = if input.x_{major}_{minor} >= 0 {{tmp_1}} else {{gate.neg(ctx, tmp_1)}};"
        ]

    def _build_ReadFloatOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ReadFloatOp)
        major: int = stmt.operator.major
        minor: int = stmt.operator.minor
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = ctx.load_witness(fixed_point_chip.quantization(input.x_{major}_{minor}));"
        ]

    def _build_ConstantOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ConstantOp)
        constant_val = stmt.operator.value
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = Constant(F::from({constant_val}));" if constant_val >= 0 else f"{{gate.neg(ctx, Constant(F::from({constant_val})))}};"
        ]

    def _build_ConstantFloatOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, ConstantFloatOp)
        constant_val = stmt.operator.value
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = Constant(fixed_point_chip.quantization({constant_val}));"
        ]

    def _build_FloatOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, FloatOp)
        x = self._get_var_name(stmt.arguments['x'])
        return [
            f"let tmp_1 = {x}.value().get_lower_128();",
            f"let tmp_2 = gate.neg(ctx, {x});"
            f"let tmp_3 = tmp_2.value().get_lower_128();",
            f"let tmp_4 = range_chip.is_less_than(ctx, {x}, Constant(F::from(0), 128);",
            f"let tmp_5 = tmp_4.value().get_lower_128() != 0;",
            f"let tmp_6 = if tmp_5 {{ctx.load_witness(fixed_point_chip.quantization(-(tmp_3 as f64)))}} else {{ctx.load_witness(fixed_point_chip.quantization(tmp_1 as f64))}};",
            f"let {self._get_var_name(stmt.stmt_id)} = tmp_6;"
        ]

    def _build_IntOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, IntOp)
        x = self._get_var_name(stmt.arguments['x'])
        return [f"let {self._get_var_name(stmt.stmt_id)} = if fixed_point_chip.dequantization({x}) >= 0 {{Constant(F::from(fixed_point_chip.dequantization({x}) as u64))}} else {{gate.neg(ctx, Constant(F::from(fixed_point_chip.dequantization({x}) as u64))))}};"]

    def _build_EqualFOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, EqualFOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let tmp_1 = fixed_point_chip.qsub(ctx, {lhs}, {rhs});",
            f"let tmp_2 = range_chip.is_less_than(ctx, tmp_1, Constant(fixed_point_chip.quantization(0.001)), 128);",
            f"let tmp_3 = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), tmp_1, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.and(ctx, tmp_2, tmp_3);"
        ]

    def _build_NotEqualFOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, NotEqualFOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let tmp_1 = fixed_point_chip.qsub(ctx, {lhs}, {rhs});",
            f"let tmp_2 = range_chip.is_less_than(ctx, tmp_1, Constant(fixed_point_chip.quantization(-0.001)), 128);",
            f"let tmp_3 = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(0.001)), tmp_1, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.or(ctx, tmp_2, tmp_3);"
        ]

    def _build_LessThanFOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LessThanFOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [f"let {self._get_var_name(stmt.stmt_id)} = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);"]

    def _build_LessThanOrEqualFOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LessThanOrEqualFOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let tmp_2 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.or(ctx, tmp_1, tmp_2);"
        ]

    def _build_GreaterThanFOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, GreaterThanFOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let tmp_2 = gate.not(ctx, tmp_1);",
            f"let tmp_3 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let tmp_4 = gate.not(ctx, tmp_3);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.and(ctx, tmp_2, tmp_4);"
        ]

    def _build_GreaterThanOrEqualFOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, GreaterThanOrEqualFOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_EqualIOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, EqualIOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [f"let {self._get_var_name(stmt.stmt_id)} = gate.is_equal(ctx, {lhs}, {rhs});"]

    def _build_NotEqualIOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, NotEqualIOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let tmp_1 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_LessThanIOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LessThanIOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [f"let {self._get_var_name(stmt.stmt_id)} = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);"]

    def _build_LessThanOrEqualIOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, LessThanOrEqualIOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let tmp_2 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.or(ctx, tmp_1, tmp_2);"
        ]

    def _build_GreaterThanIOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, GreaterThanIOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let tmp_2 = gate.not(ctx, tmp_1);",
            f"let tmp_3 = gate.is_equal(ctx, {lhs}, {rhs});",
            f"let tmp_4 = gate.not(ctx, tmp_3);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.and(ctx, tmp_2, tmp_4);"
        ]

    def _build_GreaterThanOrEqualIOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, GreaterThanOrEqualIOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let tmp_1 = range_chip.is_less_than(ctx, {lhs}, {rhs}, 128);",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_BoolCastOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, BoolCastOp)
        x = self._get_var_name(stmt.arguments['x'])
        return [
            f"let tmp_1 = gate.is_equal(ctx, {x}, Constant(F::ZERO)));",
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, tmp_1);"
        ]

    def _build_NotOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, NotOp)
        x = self._get_var_name(stmt.arguments['x'])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.not(ctx, {x});"
        ]

    def _build_AndOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, AndOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.and(ctx, {lhs}, {rhs});"
        ]

    def _build_OrOp(self, stmt: IRStatement) -> List[str]:
        assert isinstance(stmt.operator, OrOp)
        lhs = self._get_var_name(stmt.arguments['lhs'])
        rhs = self._get_var_name(stmt.arguments['rhs'])
        return [
            f"let {self._get_var_name(stmt.stmt_id)} = gate.or(ctx, {lhs}, {rhs});"
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
