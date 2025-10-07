use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
use halo2_base::gates::{GateChip, GateInstructions};
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
    pub data: Vec<f64>,
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
    let _fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let _poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    let bin_size = 3usize;
    let n = input.data.len();
    let trimmed_len = (n / bin_size) * bin_size;
    let nbins = trimmed_len / bin_size;

    // --- Load data witnesses ---
    let mut vals = Vec::new();
    for i in 0..trimmed_len {
        vals.push(ctx.load_witness(F::from_f64(input.data[i])));
    }

    // --- Compute per-bin means ---
    let mut computed_means = Vec::new();
    for b in 0..nbins {
        let mut acc = ctx.load_constant(F::ZERO);
        for j in 0..bin_size {
            let idx = b * bin_size + j;
            acc = gate.add(ctx, acc, vals[idx]);
        }
        // divide by bin_size (convert integer 3 to field element)
        let divisor = Constant(F::from(bin_size as u64));
        let mean = gate.div(ctx, acc, divisor);
        computed_means.push(mean);
    }

    // --- Check equality to provided results ---
    for k in 0..nbins {
        let expected = ctx.load_witness(F::from_f64(input.result[k]));
        let eq = gate.is_equal(ctx, computed_means[k], expected);
        gate.assert_is_const(ctx, &eq, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
