use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::poseidon::hasher::PoseidonHasher;
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
use snark_verifier_sdk::halo2::OptimizedPoseidonSpec;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub x_0: String,
    pub x_1: String,
    pub x_2: String,
    pub x_3: String,
    pub x_4: String,
    pub x_5: String,
    pub x_6: String,
    pub x_7: String,
    pub x_8: String,
    pub x_9: String,
    pub hash: String
}

fn verify_solution<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where  F: BigPrimeField {
    const PRECISION: u32 = 63;
    println!("build_lookup_bit: {:?}", builder.lookup_bits());
    let gate = GateChip::<F>::default();
    let range_chip = builder.range_chip();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let mut poseidon_hasher = PoseidonHasher::<F, T, RATE>::new(OptimizedPoseidonSpec::new::<R_F, R_P, 0>());
    let ctx = builder.main(0);
    poseidon_hasher.initialize_consts(ctx, &gate);
    let x_0 = ctx.load_witness(F::from_str_vartime(&input.x_0).expect("deserialize field element should not fail"));
    let x_1 = ctx.load_witness(F::from_str_vartime(&input.x_1).expect("deserialize field element should not fail"));
    let x_2 = ctx.load_witness(F::from_str_vartime(&input.x_2).expect("deserialize field element should not fail"));
    let x_3 = ctx.load_witness(F::from_str_vartime(&input.x_3).expect("deserialize field element should not fail"));
    let x_4 = ctx.load_witness(F::from_str_vartime(&input.x_4).expect("deserialize field element should not fail"));
    let x_5 = ctx.load_witness(F::from_str_vartime(&input.x_5).expect("deserialize field element should not fail"));
    let x_6 = ctx.load_witness(F::from_str_vartime(&input.x_6).expect("deserialize field element should not fail"));
    let x_7 = ctx.load_witness(F::from_str_vartime(&input.x_7).expect("deserialize field element should not fail"));
    let x_8 = ctx.load_witness(F::from_str_vartime(&input.x_8).expect("deserialize field element should not fail"));
    let x_9 = ctx.load_witness(F::from_str_vartime(&input.x_9).expect("deserialize field element should not fail"));
    let hash = ctx.load_witness(F::from_str_vartime(&input.hash).expect("deserialize field element should not fail"));
    make_public.push(hash);
    let my_hash = poseidon_hasher.hash_fix_len_array(ctx, &gate, &[
        x_0,
        x_1,
        x_2,
        x_3,
        x_4,
        x_5,
        x_6,
        x_7,
        x_8,
        x_9,
    ]);
    let y_13 = gate.is_equal(ctx, my_hash, hash);
    gate.assert_is_const(ctx, &y_13, &F::ONE);
    let tmp = gate.add(ctx, x_0, x_1);
    let tmp = gate.add(ctx, tmp, x_2);
    let tmp = gate.add(ctx, tmp, x_3);
    let tmp = gate.add(ctx, tmp, x_4);
    let tmp = gate.add(ctx, tmp, x_5);
    let tmp = gate.add(ctx, tmp, x_6);
    let tmp = gate.add(ctx, tmp, x_7);
    let tmp = gate.add(ctx, tmp, x_8);
    let tmp = gate.add(ctx, tmp, x_9);
    let the_expected_sum = Constant(F::from(55));
    let y_25 = gate.is_equal(ctx, tmp, the_expected_sum);
    gate.assert_is_const(ctx, &y_25, &F::ONE);
}
fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}