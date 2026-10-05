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
    pub msg: Vec<u64>,
    pub expected: String,
}

// ---------------------------------------------------------
// Zinnia Chips → Halo2 Components
// ---------------------------------------------------------

fn mimc_permute<F: ScalarField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    mut x: AssignedValue<F>,
) -> AssignedValue<F> {
    let consts = [1u64, 2, 3, 4, 5, 6, 7, 8];
    for c in consts {
        let c_val = Constant(F::from(c));
        let t = gate.add(ctx, x, c_val);
        let t2 = gate.mul(ctx, t, t);
        let t3 = gate.mul(ctx, t2, t);
        x = t3;
    }
    x
}

fn mimc3_hash_3<F: ScalarField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    msg: &Vec<AssignedValue<F>>,
) -> AssignedValue<F> {
    let mut state = ctx.load_constant(F::ZERO);
    for i in 0..3 {
        let s = gate.add(ctx, state, msg[i]);
        state = mimc_permute(ctx, gate, s);
    }
    state
}

// ---------------------------------------------------------
// Main verification circuit
// ---------------------------------------------------------

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
    let mut poseidon_hasher =
        PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);

    // Load inputs
    let msg: Vec<AssignedValue<F>> = input
        .msg
        .iter()
        .map(|x| ctx.load_witness(F::from(*x)))
        .collect();
    let expected = ctx.load_witness(F::from_str_vartime(&input.expected).unwrap());

    // Compute MiMC-3 hash
    let h = mimc3_hash_3(ctx, &gate, &msg);

    // Equality check
    let eq = gate.is_equal(ctx, h, expected);
    gate.assert_is_const(ctx, &eq, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
