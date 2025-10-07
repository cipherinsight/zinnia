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
    pub x: Vec<f64>,
    pub result: Vec<f64>,
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    _make_public: &mut Vec<AssignedValue<F>>,
)
where
    F: BigPrimeField,
{
    const PRECISION: u32 = 63;
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let _fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    let n = input.x.len();
    let m = input.result.len();

    // --- Load x as witnesses ---
    let mut x_vals = Vec::new();
    for i in 0..n {
        x_vals.push(ctx.load_witness(F::from_f64(input.x[i])));
    }

    // --- Filtering: build expected non-negative elements in-circuit ---
    let mut expected = Vec::new();
    for i in 0..n {
        let xi = x_vals[i];
        // predicate: xi >= 0  → !(xi < 0)
        let zero = Constant(F::ZERO);
        let lt0 = range_chip.is_less_than(ctx, xi, zero, 128); // xi < 0
        let keep_flag = gate.not(ctx, lt0);
        // next value = xi * keep_flag
        let kept = gate.mul(ctx, xi, keep_flag);
        expected.push((kept, keep_flag));
    }

    // Count number of nonnegative entries to check shape
    let mut count = ctx.load_constant(F::ZERO);
    for (_, flag) in &expected {
        count = gate.add(ctx, count, *flag);
    }

    gate.assert_is_const(ctx, &count, &F::from(m as u64));

    // --- Construct compacted output ---
    // expected_result[k] = the k-th kept element
    let mut compact = vec![ctx.load_constant(F::ZERO); m];
    let mut prefix = ctx.load_constant(F::ZERO);

    for (xi, flag) in expected {
        for k in 0..m {
            let is_pos = gate.is_equal(ctx, prefix, Constant(F::from(k as u64)));
            let sel = gate.and(ctx, flag, is_pos);
            compact[k] = gate.add(ctx, compact[k], gate.mul(ctx, xi, sel));
        }
        prefix = gate.add(ctx, prefix, flag);
    }

    // --- Load result witnesses and assert equality ---
    for k in 0..m {
        let exp = compact[k];
        let res = ctx.load_witness(F::from_f64(input.result[k]));
        let eq = gate.is_equal(ctx, exp, res);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
