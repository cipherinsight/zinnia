use std::env::var;
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
use snark_verifier_sdk::snark_verifier::halo2_ecc::bigint::negative;
use snark_verifier_sdk::snark_verifier::loader::halo2::IntegerInstructions;

const T: usize = 3;
const RATE: usize = 2;
const R_F: usize = 8;
const R_P: usize = 57;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub data: Vec<f64>,
    pub centroids: Vec<f64>,
    pub classifications: Vec<u128>,
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

    // load variables
    let classifications = input.classifications.iter().map(|x| ctx.load_witness(F::from_u128(*x))).collect::<Vec<_>>();
    let mut centroids = input.centroids.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();
    let data = input.data.iter().map(|x| ctx.load_witness(fixed_point_chip.quantization(*x))).collect::<Vec<_>>();

    // do computations
    let n = 10;
    let d = 2;
    let classes = 3;
    let mut labels = vec![ctx.load_constant(F::ZERO); n];
    for _ in 0..10 {
        for i in 0..n {
            let mut dists = vec![ctx.load_constant(F::ZERO), ctx.load_constant(F::ZERO), ctx.load_constant(F::ZERO)];
            for j in 0..classes {
                let mut diff = ctx.load_constant(fixed_point_chip.quantization(0.0));
                for k in 0..d {
                    let diff_ = fixed_point_chip.qsub(ctx, data[i * d + k], centroids[j * d + k]);
                    let diff_sq = fixed_point_chip.qmul(ctx, diff_, diff_);
                    diff = fixed_point_chip.qadd(ctx, diff, diff_sq);
                }
                dists[j] = diff;
            }
            let mut new_label = ctx.load_constant(F::ZERO);
            let mut min_dist = dists[0];
            for j in 1..classes {
                let is_less = range_chip.is_less_than(ctx, dists[j], min_dist, 128);
                min_dist = gate.select(ctx, dists[j], min_dist, is_less);
                let tmp = ctx.load_constant(F::from_u128(j as u128));
                new_label = gate.select(ctx, tmp, new_label, is_less);
            }
            labels[i] = new_label;
        }

        let mut new_centroids = vec![ctx.load_constant(fixed_point_chip.quantization(0.0)); classes * d];
        let mut counts = vec![ctx.load_constant(fixed_point_chip.quantization(0.0)); classes];
        for i in 0..n {
            let label_i = labels[i];
            for j in 0..classes {
                let j_eq_label_i = gate.is_equal(ctx, label_i, Constant(F::from_u128(j as u128)));
                for k in 0..d {
                    let new_centroid = fixed_point_chip.qadd(ctx, new_centroids[j * d + k], data[i * d + k]);
                    new_centroids[j * d + k] = gate.select(ctx, new_centroid, new_centroids[j * d + k], j_eq_label_i);
                }
                let new_count = fixed_point_chip.qadd(ctx, counts[j], Constant(fixed_point_chip.quantization(1.0)));
                counts[j] = gate.select(ctx, new_count, counts[j], j_eq_label_i);
            }
        }
        for i in 0..classes {
            for k in 0..d {
                let new_value = fixed_point_chip.qdiv(ctx, new_centroids[i * d + k], counts[i]);
                new_centroids[i * d + k] = new_value;
            }
        }
        centroids = new_centroids;
    }
    for i in 0..classes {
        let correct = gate.is_equal(ctx, labels[i],classifications[i]);
        gate.assert_is_const(ctx, &correct, &F::ONE);
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
