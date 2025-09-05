use std::result;

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
    pub data: Vec<f64>,
    pub power: f64,
    pub results: Vec<f64>,
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

    // load data
    let data = input.data.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();
    let result = input.results.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();
    let power = ctx.load_witness(fixed_point_chip.quantization(input.power));
    for i in 0..(8*8) {
        let input_lt_0 = range_chip.is_less_than(ctx, data[i], Constant(fixed_point_chip.quantization(0.0)), 128);
        let input_gte_0 = gate.not(ctx, input_lt_0);
        gate.assert_is_const(ctx, &input_gte_0, &F::ONE);
        let res = fixed_point_chip.qpow(ctx, data[i], power);
        let res_loss = fixed_point_chip.qsub(ctx, res, result[i]);
        let lower_bound = range_chip.is_less_than(ctx, res_loss, Constant(fixed_point_chip.quantization(0.001)), 128);
        let upper_bound = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), res_loss, 128);
        let valid = gate.and(ctx, lower_bound, upper_bound);
        gate.assert_is_const(ctx, &valid, &F::ONE);
    }

}
fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
