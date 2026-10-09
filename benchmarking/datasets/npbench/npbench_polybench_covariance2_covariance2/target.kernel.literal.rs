// npbench_polybench_covariance2_covariance2__kernel: literal translation of kernel.py (LITERAL_BASELINES.md).
// np.cov uses means, full centered matrix, full dot matrix and division by NO-1; no symmetry reuse.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const NO: usize = 16; // = data_0
const NV: usize = 14; // = data_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let m = inputs.int("M")?;
    let data = inputs.floats("data", &[NO, NV])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    // Precondition 0 <= M <= 1024: M and 1024 - M in [0, 2^16).
    let m = q.load_int(ctx,m);
    q.range.range_check(ctx, m, 16);
    let headroom = q.sub(ctx, Constant(fe(1024)), m);
    q.range.range_check(ctx, headroom, 16);

    let data = q.load_floats(ctx, &data);
    let mut transposed = vec![vec![ctx.load_zero(); NO]; NV];
    for v in 0..NV { for o in 0..NO { transposed[v][o] = data[o*NV+v]; } }
    let mut means = Vec::new();
    for v in 0..NV {
        let mut total = ctx.load_zero();
        for o in 0..NO { total = q.add(ctx,total,transposed[v][o]); }
        means.push(q.floor_div_const(ctx,total,NO as u64).0.cell);
    }
    let mut deviations = transposed.clone();
    for v in 0..NV { for o in 0..NO { deviations[v][o] = q.sub(ctx,transposed[v][o],means[v]); } }
    let mut products = vec![ctx.load_zero(); NV*NV];
    for a in 0..NV { for b in 0..NV {
        let mut raw = ctx.load_zero();
        for o in 0..NO { let term = q.gate.mul(ctx,deviations[a][o],deviations[b][o]); raw = q.add(ctx,raw,term); }
        products[a*NV+b] = q.rescale(ctx,raw).cell;
    } }
    let cov: Vec<Cell> = products.iter().map(|v| q.floor_div_const(ctx,*v,(NO-1) as u64).0.cell).collect();
    out.floats(&cov); Ok(())
}
