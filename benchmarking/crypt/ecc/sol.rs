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
    pub x1: String,
    pub y1: String,
    pub x2: String,
    pub y2: String,
    pub x3: String,
    pub y3: String
}

fn baby_jubjub_ecc<F: ScalarField>(
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

    let x1 = ctx.load_witness(F::from_str_vartime(&input.x1).expect("deserialize field element should not fail"));
    let y1 = ctx.load_witness(F::from_str_vartime(&input.y1).expect("deserialize field element should not fail"));
    let x2 = ctx.load_witness(F::from_str_vartime(&input.x2).expect("deserialize field element should not fail"));
    let y2 = ctx.load_witness(F::from_str_vartime(&input.y2).expect("deserialize field element should not fail"));
    let x3 = ctx.load_witness(F::from_str_vartime(&input.x3).expect("deserialize field element should not fail"));
    let y3 = ctx.load_witness(F::from_str_vartime(&input.y3).expect("deserialize field element should not fail"));

    let const_a = ctx.load_constant(F::from(168700));
    let const_d = ctx.load_constant(F::from(168696));

    // check point 1 should be on the curve
    let x1_square = gate.mul(ctx, x1, x1);
    let y1_square = gate.mul(ctx, y1, y1);
    let left = gate.mul(ctx, x1_square, const_a);
    let left = gate.add(ctx, left, y1_square);
    let right = gate.mul(ctx, x1_square, y1_square);
    let right = gate.mul(ctx, right, const_d);
    let right = gate.add(ctx, right, Constant(F::ONE));
    let is_on_curve_1 = gate.is_equal(ctx, left, right);
    gate.assert_is_const(ctx, &is_on_curve_1, &F::ONE);

    // check point 2 should be on the curve
    let x2_square = gate.mul(ctx, x2, x2);
    let y2_square = gate.mul(ctx, y2, y2);
    let left = gate.mul(ctx, x2_square, const_a);
    let left = gate.add(ctx, left, y2_square);
    let right = gate.mul(ctx, x2_square, y2_square);
    let right = gate.mul(ctx, right, const_d);
    let right = gate.add(ctx, right, Constant(F::ONE));
    let is_on_curve_2 = gate.is_equal(ctx, left, right);
    gate.assert_is_const(ctx, &is_on_curve_2, &F::ONE);

    // check point 3 should be on the curve
    let x3_square = gate.mul(ctx, x3, x3);
    let y3_square = gate.mul(ctx, y3, y3);
    let left = gate.mul(ctx, x3_square, const_a);
    let left = gate.add(ctx, left, y3_square);
    let right = gate.mul(ctx, x3_square, y3_square);
    let right = gate.mul(ctx, right, const_d);
    let right = gate.add(ctx, right, Constant(F::ONE));
    let is_on_curve_3 = gate.is_equal(ctx, left, right);
    gate.assert_is_const(ctx, &is_on_curve_3, &F::ONE);

    // calculate the added point
    let beta = gate.mul(ctx, x1, y2);
    let gamma = gate.mul(ctx, y1, x2);
    let tmp1 = gate.add(ctx, x2, y2);
    let tmp2 = gate.sub(ctx, Constant(F::ZERO), const_a);
    let tmp3 = gate.mul(ctx, tmp2, x1);
    let tmp4 = gate.add(ctx, tmp3, y1);
    let delta = gate.mul(ctx, tmp1, tmp4);
    let tau = gate.mul(ctx, beta, gamma);
    let tmp5 = gate.add(ctx, beta, gamma);
    let tmp6 = gate.mul(ctx, const_d, tau);
    let tmp7 = gate.add(ctx, Constant(F::ONE), tmp6);
    let inv_of_tmp7 = ctx.load_witness(tmp7.value.evaluate().invert().unwrap());
    let tmp8 = gate.mul(ctx, tmp7, inv_of_tmp7);
    let tmp8_eq_1 = gate.is_equal(ctx, tmp8, Constant(F::ONE));
    gate.assert_is_const(ctx, &tmp8_eq_1, &F::ONE);
    let x4 = gate.mul(ctx, tmp5, inv_of_tmp7);
    let tmp9 = gate.mul(ctx, const_d, tau);
    let tmp10 = gate.sub(ctx, Constant(F::ONE), tmp9);
    let inv_of_tmp10 = ctx.load_witness(tmp10.value.evaluate().invert().unwrap());
    let tmp11 = gate.mul(ctx, tmp10, inv_of_tmp10);
    let tmp11_eq_1 = gate.is_equal(ctx, tmp11, Constant(F::ONE));
    gate.assert_is_const(ctx, &tmp11_eq_1, &F::ONE);
    let tmp12 = gate.mul(ctx, const_a, beta);
    let tmp13 = gate.add(ctx, tmp12, delta);
    let tmp14 = gate.sub(ctx, tmp13, gamma);
    let y4 = gate.mul(ctx, tmp14, inv_of_tmp10);

    // check point 4 should be equal to point 3
    let x4_eq_x3 = gate.is_equal(ctx, x4, x3);
    gate.assert_is_const(ctx, &x4_eq_x3, &F::ONE);
    let y4_eq_y3 = gate.is_equal(ctx, y4, y3);
    gate.assert_is_const(ctx, &y4_eq_y3, &F::ONE);
}
fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(baby_jubjub_ecc, args);
}