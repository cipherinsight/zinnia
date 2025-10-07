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
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub c: Vec<u64>,
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

    // Load vectors
    let a: Vec<AssignedValue<F>> = input.a.iter().map(|x| ctx.load_witness(F::from(*x))).collect();
    let b: Vec<AssignedValue<F>> = input.b.iter().map(|x| ctx.load_witness(F::from(*x))).collect();
    let c: Vec<AssignedValue<F>> = input.c.iter().map(|x| ctx.load_witness(F::from(*x))).collect();

    let result: Vec<AssignedValue<F>> = input
        .result
        .iter()
        .map(|x| ctx.load_witness(fixed_point_chip.quantization(*x)))
        .collect();

    // For each index j, compute mean = (a[j] + b[j] + c[j]) / 3
    let const3 = Constant(fixed_point_chip.quantization(3.0));
    for j in 0..a.len() {
        let a_f = fixed_point_chip.qcast_int(ctx, a[j]);
        let b_f = fixed_point_chip.qcast_int(ctx, b[j]);
        let c_f = fixed_point_chip.qcast_int(ctx, c[j]);
        let sum = fixed_point_chip.qadd(ctx, fixed_point_chip.qadd(ctx, a_f, b_f), c_f);
        let mean = fixed_point_chip.qdiv(ctx, sum, const3);

        // Check |mean - result[j]| ≤ 1e-3
        let diff = fixed_point_chip.qsub(ctx, mean, result[j]);
        let tol = Constant(fixed_point_chip.quantization(0.001));
        let upper = range_chip.is_less_than(ctx, diff, tol, 128);
        let lower = range_chip.is_less_than(ctx, Constant(fixed_point_chip.quantization(-0.001)), diff, 128);
        let ok = gate.and(ctx, upper, lower);
        gate.assert_is_const(ctx, &ok, &F::ONE);
    }
}

// Helper trait extension: integer to fixed-point conversion
trait FixedPointCastExt<F: BigPrimeField> {
    fn qcast_int(
        &self,
        ctx: &mut Context<F>,
        int_val: AssignedValue<F>,
    ) -> AssignedValue<F>;
}
impl<F: BigPrimeField> FixedPointCastExt<F> for FixedPointChip<F, 63> {
    fn qcast_int(
        &self,
        ctx: &mut Context<F>,
        int_val: AssignedValue<F>,
    ) -> AssignedValue<F> {
        let val = self.dequantization(*int_val.value());
        let quant = self.quantization(val);
        ctx.load_witness(quant)
    }
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(verify_solution, args);
}
