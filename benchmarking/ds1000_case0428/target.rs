// DS-1000 case 428 — Halo2 baseline.
// Kernel: `(a.mean() - 3*a.std(), a.mean() + 3*a.std())` on Float[30].
// High-level reimplementation: mean = sum/N, std = sqrt(variance), output (mean-3σ, mean+3σ).
// kernel.py recomputes a.std() inside the tuple expression so the second
// call is emitted explicitly — preserved here.

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)]
use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput {
    pub a: Vec<f64>,
}

fn mean_std_bracket<F: ScalarField>(
    builder: &mut BaseCircuitBuilder<F>,
    input: CircuitInput,
    make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const PRECISION: u32 = 32;
    let fpc = FixedPointChip::<F, PRECISION>::default(builder);
    let ctx = builder.main(0);

    assert_eq!(input.a.len(), 30);
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v)))
        .collect();

    // mean = sum / N
    let mut sum = cells[0];
    for k in 1..30 {
        sum = fpc.qadd(ctx, sum, cells[k]);
    }
    let n = ctx.load_witness(fpc.quantization(30.0));
    let mean = fpc.qdiv(ctx, sum, n);

    // std = sqrt( sum((x - mean)^2) / N ).
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

    let std1 = compute_std(&fpc, ctx);
    let std2 = compute_std(&fpc, ctx);

    let three = ctx.load_witness(fpc.quantization(3.0));
    let three_std1 = fpc.qmul(ctx, three, std1);
    let three_std2 = fpc.qmul(ctx, three, std2);

    let lo = fpc.qsub(ctx, mean, three_std1);
    let hi = fpc.qadd(ctx, mean, three_std2);

    make_public.push(lo);
    make_public.push(hi);
}

fn main() {
    env_logger::init();
    let args = Cli::parse();
    run(mean_std_bracket, args);
}
