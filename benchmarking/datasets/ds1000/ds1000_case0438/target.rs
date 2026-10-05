// DS-1000 case 438 — Halo2 baseline. `np.corrcoef(post, distance)[0][1]`
// on Integer[4] × Integer[4]. High-level reimplementation: Pearson r.
//   r = cov(x, y) / (σ_x · σ_y), with σ_x · σ_y = sqrt(varx · vary).

use clap::Parser;
use halo2_base::utils::{ScalarField, BigPrimeField};
use halo2_graph::gadget::fixed_point::{FixedPointChip, FixedPointInstructions};
use halo2_base::gates::circuit::builder::BaseCircuitBuilder;
use serde::{Serialize, Deserialize};
use halo2_base::AssignedValue;
#[allow(unused_imports)] use halo2_graph::scaffold::cmd::Cli;
use halo2_graph::scaffold::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitInput { pub post: Vec<String>, pub distance: Vec<String> }

fn corrcoef<F: ScalarField>(
    b: &mut BaseCircuitBuilder<F>, input: CircuitInput, make_public: &mut Vec<AssignedValue<F>>,
) where F: BigPrimeField {
    const P: u32 = 32;
    let fpc = FixedPointChip::<F, P>::default(b);
    let ctx = b.main(0);
    assert_eq!(input.post.len(), 4);
    assert_eq!(input.distance.len(), 4);
    let load_fp = |ctx: &mut halo2_base::Context<F>, s: &str| -> AssignedValue<F> {
        let v: i64 = s.parse().unwrap();
        ctx.load_witness(fpc.quantization(v as f64))
    };
    let xs: Vec<AssignedValue<F>> = input.post.iter().map(|s| load_fp(ctx, s)).collect();
    let ys: Vec<AssignedValue<F>> = input.distance.iter().map(|s| load_fp(ctx, s)).collect();
    let n = ctx.load_witness(fpc.quantization(4.0));
    let mean_of = |ctx: &mut halo2_base::Context<F>, vs: &[AssignedValue<F>]| {
        let mut s = vs[0];
        for k in 1..vs.len() { s = fpc.qadd(ctx, s, vs[k]); }
        fpc.qdiv(ctx, s, n)
    };
    let xbar = mean_of(ctx, &xs);
    let ybar = mean_of(ctx, &ys);
    let mut cov = { let dx = fpc.qsub(ctx, xs[0], xbar); let dy = fpc.qsub(ctx, ys[0], ybar); fpc.qmul(ctx, dx, dy) };
    let mut vx = { let dx = fpc.qsub(ctx, xs[0], xbar); fpc.qmul(ctx, dx, dx) };
    let mut vy = { let dy = fpc.qsub(ctx, ys[0], ybar); fpc.qmul(ctx, dy, dy) };
    for k in 1..4 {
        let dx = fpc.qsub(ctx, xs[k], xbar);
        let dy = fpc.qsub(ctx, ys[k], ybar);
        let cov_k = fpc.qmul(ctx, dx, dy);
        let vx_k = fpc.qmul(ctx, dx, dx);
        let vy_k = fpc.qmul(ctx, dy, dy);
        cov = fpc.qadd(ctx, cov, cov_k);
        vx = fpc.qadd(ctx, vx, vx_k);
        vy = fpc.qadd(ctx, vy, vy_k);
    }
    let prod_v = fpc.qmul(ctx, vx, vy);
    let denom = fpc.qsqrt(ctx, prod_v);
    let r = fpc.qdiv(ctx, cov, denom);
    make_public.push(r);
}
fn main() { env_logger::init(); let args = Cli::parse(); run(corrcoef, args); }
