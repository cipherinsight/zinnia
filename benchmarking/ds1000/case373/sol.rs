use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::{Constant, Witness},
};
use halo2_graph::gadget::fixed_point::FixedPointChip;
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;
use serde::{Serialize, Deserialize};
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub grades: Vec<f64>,
    pub result: Vec<f64>,
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
)
where
    F: BigPrimeField,
{
    const PRECISION: u32 = 63; // same quantization scale as IR spec
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    let n = input.grades.len();
    assert_eq!(n, 27);

    // ---- Load grades ----
    let mut grades_fp: Vec<AssignedValue<F>> = Vec::new();
    for g in input.grades.iter() {
        let q = fixed_point_chip.quantization(*g);
        grades_fp.push(ctx.load_witness(q));
    }

    // ---- Load expected ECDF result ----
    let mut result_fp: Vec<AssignedValue<F>> = Vec::new();
    for r in input.result.iter() {
        let q = fixed_point_chip.quantization(*r);
        result_fp.push(ctx.load_witness(q));
    }

    // ---- Step 1: verify non-decreasing ----
    for i in 0..(n - 1) {
        let le = range_chip.is_less_than(ctx, grades_fp[i], grades_fp[i + 1], 128);
        let ge = gate.is_equal(ctx, grades_fp[i], grades_fp[i + 1]);
        let le_or_eq = gate.or(ctx, le, ge);
        gate.assert_is_const(ctx, &le_or_eq, &F::ONE);
    }

    // ---- Step 2 & 3: verify ECDF values (i+1)/n ----
    for i in 0..n {
        let ideal_val = fixed_point_chip.quantization((i as f64 + 1.0) / n as f64);
        let ideal_fp = ctx.load_constant(ideal_val);
        let eq = gate.is_equal(ctx, result_fp[i], ideal_fp);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
