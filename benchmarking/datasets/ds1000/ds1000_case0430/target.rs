// DS-1000 case 430 — Halo2 baseline. (mean - 3·std, mean + 3·std) on Float[30].
// Same shape as 0428. Uses witness-and-verify sqrt.
use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use halo2_base::gates::{GateChip, GateInstructions};
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub a: Vec<f64> }

fn mean3sigma<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let gate = GateChip::<F>::default();
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.a.len(), 30);
    let cells: Vec<AssignedValue<F>> = input.a.iter()
        .map(|v| ctx.load_witness(fpc.quantization(*v))).collect();
    let mut sum = cells[0];
    for k in 1..30 { sum = fpc.qadd(ctx, sum, cells[k]); }
    let n = ctx.load_witness(fpc.quantization(30.0));
    let mean = fpc.qdiv(ctx, sum, n);

    let compute_std = |fpc: &FixedPointChip<F, P>, ctx: &mut halo2_base::Context<F>| -> AssignedValue<F> {
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
    let s1 = compute_std(&fpc, ctx);
    let s2 = compute_std(&fpc, ctx);
    let ctx = b.main(0);
    let three = ctx.load_witness(fpc.quantization(3.0));
    let ts1 = fpc.qmul(ctx, three, s1);
    let ts2 = fpc.qmul(ctx, three, s2);
    let lo = fpc.qsub(ctx, mean, ts1);
    let hi = fpc.qadd(ctx, mean, ts2);
    make_public.push(lo);
    make_public.push(hi);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(mean3sigma, args); }
