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
    pub g: String,
    pub sk: String,
    pub r: String,
    pub msg: String,
}

// ---------------------------------------------------------
// Zinnia Chips → Halo2 reusable components
// ---------------------------------------------------------

fn elgamal_keygen<F: ScalarField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    g: AssignedValue<F>,
    sk: AssignedValue<F>,
) -> AssignedValue<F> {
    // pk = g ^ sk
    gate.pow_var(ctx, g, sk, 251)
}

fn elgamal_encrypt<F: ScalarField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    g: AssignedValue<F>,
    pk: AssignedValue<F>,
    msg: AssignedValue<F>,
    r: AssignedValue<F>,
) -> (AssignedValue<F>, AssignedValue<F>) {
    // c1 = g^r
    let c1 = gate.pow_var(ctx, g, r, 251);
    // c2 = msg * pk^r
    let pk_r = gate.pow_var(ctx, pk, r, 251);
    let c2 = gate.mul(ctx, msg, pk_r);
    (c1, c2)
}

fn elgamal_decrypt<F: ScalarField>(
    ctx: &mut Context<F>,
    gate: &GateChip<F>,
    sk: AssignedValue<F>,
    c1: AssignedValue<F>,
    c2: AssignedValue<F>,
) -> AssignedValue<F> {
    // shared = c1 ^ sk
    let shared = gate.pow_var(ctx, c1, sk, 251);
    // msg = c2 * inv(shared)
    let inv_shared = gate.inv(ctx, shared);
    gate.mul(ctx, c2, inv_shared)
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

    // Parse big integers (string → field)
    let g = ctx.load_witness(F::from_str_vartime(&input.g).unwrap());
    let sk = ctx.load_witness(F::from_str_vartime(&input.sk).unwrap());
    let r = ctx.load_witness(F::from_str_vartime(&input.r).unwrap());
    let msg = ctx.load_witness(F::from_str_vartime(&input.msg).unwrap());

    // Key generation
    let pk = elgamal_keygen(ctx, &gate, g, sk);
    // Encryption
    let (c1, c2) = elgamal_encrypt(ctx, &gate, g, pk, msg, r);
    // Decryption
    let recovered = elgamal_decrypt(ctx, &gate, sk, c1, c2);

    // Round-trip consistency check
    let eq = gate.is_equal(ctx, recovered, msg);
    gate.assert_is_const(ctx, &eq, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
