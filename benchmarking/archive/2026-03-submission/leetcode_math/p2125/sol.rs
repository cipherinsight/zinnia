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
    pub bank: Vec<u128>,
    pub expected: u128,
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
    let expected = ctx.load_witness(F::from_u128(input.expected));
    let bank = input.bank.iter().map(|x| ctx.load_witness(F::from_u128(*x))).collect::<Vec<_>>();
    // apply constraints
    let mut result = ctx.load_constant(F::ZERO);
    for si in 0..5 {
        for sj in 0..5 {
            for ti in 0..5 {
                for tj in 0..5 {
                    let bank_si_sj = bank[si * 5 + sj];
                    let bank_ti_tj = bank[ti * 5 + tj];
                    let bank_si_sj_eq_1 = gate.is_equal(ctx, bank_si_sj, Constant(F::ONE));
                    let bank_ti_tj_eq_1 = gate.is_equal(ctx, bank_ti_tj, Constant(F::ONE));
                    let si_lt_ti = range_chip.is_less_than(ctx, Constant(F::from(si as u64)), Constant(F::from(ti as u64)), 128);
                    let add_one = gate.and(ctx, bank_si_sj_eq_1, bank_ti_tj_eq_1);
                    let mut add_one = gate.and(ctx, add_one, si_lt_ti);
                    for k in (si+1)..ti {
                        let mut any_one = ctx.load_constant(F::ZERO);
                        for j in sj..tj {
                            let equals_one = gate.is_equal(ctx, bank[k * 5 + j], Constant(F::ONE));
                            any_one = gate.or(ctx, any_one, equals_one);
                        }
                        add_one = gate.select(ctx, Constant(F::ZERO), add_one, any_one);
                    }
                    result = gate.add(ctx, result, add_one);
                }
            }
        }
    }
    let result_eq_expected = gate.is_equal(ctx, result, expected);
    gate.assert_is_const(ctx, &result_eq_expected, &F::ONE);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
