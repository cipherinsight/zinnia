use std::result;
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
use halo2_base::gates::{GateChip, GateInstructions, RangeInstructions};
use halo2_base::{
    Context,
    AssignedValue,
    QuantumCell::{Constant, Existing, Witness},
};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
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
    pub eval: Vec<f64>,
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
    const PRECISION: u32 = 63;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    let n = 27;
    let m = 3;

    // Load inputs
    let grades: Vec<AssignedValue<F>> = input
        .grades
        .iter()
        .map(|x| ctx.load_witness(fixed_point_chip.quantization(*x)))
        .collect();
    let evals: Vec<AssignedValue<F>> = input
        .eval
        .iter()
        .map(|x| ctx.load_witness(fixed_point_chip.quantization(*x)))
        .collect();
    let results: Vec<AssignedValue<F>> = input
        .result
        .iter()
        .map(|x| ctx.load_witness(fixed_point_chip.quantization(*x)))
        .collect();

    // 1) Verify non-decreasing: grades[i] <= grades[i+1]
    for i in 0..(n - 1) {
        let strict_gt = range_chip.is_less_than(ctx, grades[i + 1], grades[i], 128); // (i+1) < i  ?
        let le = gate.not(ctx, strict_gt); // !(grades[i+1] < grades[i])  => grades[i] <= grades[i+1]
        gate.assert_is_const(ctx, &le, &F::ONE);
    }

    // 2) ys[i] = (i+1)/n
    let n_const = Constant(fixed_point_chip.quantization(n as f64));
    let mut ys: Vec<AssignedValue<F>> = Vec::with_capacity(n);
    for i in 0..n {
        let i1 = Constant(fixed_point_chip.quantization((i + 1) as f64));
        ys.push(fixed_point_chip.qdiv(ctx, i1, n_const));
    }

    // 3) Apply ECDF to each eval[i]
    for i in 0..m {
        let x = evals[i];

        // Extremes
        let lt_first = range_chip.is_less_than(ctx, x, grades[0], 128);           // x < first
        let lt_last  = range_chip.is_less_than(ctx, x, grades[n - 1], 128);       // x < last
        let ge_last  = gate.not(ctx, lt_last);                                     // x >= last

        // val = 0; if x >= last -> 1
        let mut val = ctx.load_constant(fixed_point_chip.quantization(0.0));
        let one = ctx.load_constant(fixed_point_chip.quantization(1.0));
        val = gate.select(ctx, one, val, ge_last);

        // Find smallest j with grades[j] > x
        // j_idx initialized to (n - 1); updated only on the FIRST match
        let mut j_idx = ctx.load_constant(fixed_point_chip.quantization((n - 1) as f64));
        let mut found = ctx.load_constant(F::ZERO); // boolean 0/1

        for k in 0..n {
            let gt = range_chip.is_less_than(ctx, x, grades[k], 128); // x < grades[k] ?
            let not_found = gate.not(ctx, found);
            let take = gate.and(ctx, not_found, gt);

            let k_const = Constant(fixed_point_chip.quantization(k as f64));
            j_idx = gate.select(ctx, k_const, j_idx, take);

            found = gate.or(ctx, found, gt);
        }

        // y_sel = ys[j_idx - 1]
        let j_minus_1 = fixed_point_chip.qsub(ctx, j_idx, Constant(fixed_point_chip.quantization(1.0)));
        let mut y_sel = ctx.load_constant(fixed_point_chip.quantization(0.0));
        for k in 0..n {
            let k_const = Constant(fixed_point_chip.quantization(k as f64));
            let eq = gate.is_equal(ctx, j_minus_1, k_const);
            y_sel = gate.select(ctx, ys[k], y_sel, eq);
        }

        // Inner region: (!lt_first) & (!ge_last)
        let tmp1 = gate.not(ctx, lt_first);
        let tmp2 = gate.not(ctx, ge_last);
        let inner = gate.and(ctx, tmp1, tmp2);
        let computed = gate.select(ctx, y_sel, val, inner);

        // Debug (optional)
        // println!("Computed {:?} vs Expected {:?}", fixed_point_chip.dequantization(*computed.value()), fixed_point_chip.dequantization(*results[i].value()));

        // Assert |computed - results[i]| <= 1e-3
        let diff = fixed_point_chip.qsub(ctx, computed, results[i]);
        let le = range_chip.is_less_than(ctx, diff, Constant(fixed_point_chip.quantization(0.001)), 128);
        let ge = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), diff, 128);
        let ok = gate.and(ctx, le, ge);
        gate.assert_is_const(ctx, &ok, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
