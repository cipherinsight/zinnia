// DS-1000 case 429 — Halo2 baseline. (mean - 2·std, mean + 2·std) on Float[30].
// Same shape as 0428 but ±2σ instead of ±3σ. Uses halo2-graph's witness-and-
// verify sqrt (one qmul + ±ε check).

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<f64> }

fn mean_std_2sigma<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let gate = GateChip::<F>::default();
    let fixed_point_chip = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);
    assert_eq!(input.a.len(), 30);
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fixed_point_chip.quantization(*v)))
        .collect();
    let mut sum = cells[0];
    for k in 1..30 { sum = fixed_point_chip.qadd(ctx, sum, cells[k]); }
    let n = ctx.load_witness(fixed_point_chip.quantization(30.0));
    let mean = fixed_point_chip.qdiv(ctx, sum, n);

    let compute_std = |fpc: &FixedPointChip<F, PRECISION>, ctx: &mut halo2_base::Context<F>| -> AssignedValue<F> {
        let mut sq_sum = {
            let d = fpc.qsub(ctx, cells[0], mean);
            fpc.qmul(ctx, d, d)
        };
        for k in 1..30 {
            let d = fpc.qsub(ctx, cells[k], mean);
            let d2 = fpc.qmul(ctx, d, d);
            sq_sum = fpc.qadd(ctx, sq_sum, d2);
        }
        let variance = fpc.qdiv(ctx, sq_sum, n);
        fpc.qsqrt(ctx, variance)
    };
    let std1 = compute_std(&fixed_point_chip, ctx);
    let std2 = compute_std(&fixed_point_chip, ctx);
    let ctx = builder.main(0);
    let two = ctx.load_witness(fixed_point_chip.quantization(2.0));
    let two_std1 = fixed_point_chip.qmul(ctx, two, std1);
    let two_std2 = fixed_point_chip.qmul(ctx, two, std2);
    let lo = fixed_point_chip.qsub(ctx, mean, two_std1);
    let hi = fixed_point_chip.qadd(ctx, mean, two_std2);
    make_public.push(lo);
    make_public.push(hi);
}

fn main() { env_logger::init(); let args = Cli::parse(); run(mean_std_2sigma, args); }
