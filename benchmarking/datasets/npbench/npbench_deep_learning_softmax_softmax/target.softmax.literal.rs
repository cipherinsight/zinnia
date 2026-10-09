// npbench_deep_learning_softmax_softmax__softmax: literal translation of kernel.py (LITERAL_BASELINES.md).
// Max is an index-order pairwise scan with fresh bounds; every array call is a full pass.
// Exp uses the shared canonical family used by this case's optimized arithmetic.
use crate::prelude::*;
const N: usize = 2; // = x_0
const H: usize = 2; // = x_1
const SN: usize = 16; // = x_2
const SM: usize = 16; // = x_3
const ROWS: usize = N * H * SN;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N, H, SN, SM])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let x = q.load_floats(ctx, &x);
    let mut tmp_max = Vec::with_capacity(ROWS);
    for row in x.chunks(SM) {
        let mut best = row[0];
        for v in &row[1..] {
            let a = q.bound(ctx, best);
            let b = q.bound(ctx, *v);
            best = q.max(ctx, a, b).cell;
        }
        tmp_max.push(best);
    }
    let gaps: Vec<Cell> = (0..ROWS * SM)
        .map(|i| q.sub(ctx, x[i], tmp_max[i / SM]))
        .collect();
    let tmp_out: Vec<Cell> = gaps
        .iter()
        .map(|g| {
            let g = q.bound(ctx, *g);
            q.canonical_exp(ctx, g).cell
        })
        .collect();
    let mut tmp_sum = Vec::with_capacity(ROWS);
    for row in tmp_out.chunks(SM) {
        let mut sum = q.float_const(ctx, 0.0);
        for v in row {
            sum = q.add(ctx, sum, *v);
        }
        tmp_sum.push(sum);
    }
    let result: Vec<Cell> = (0..ROWS * SM)
        .map(|i| {
            let den = q.bound(ctx, tmp_sum[i / SM]);
            q.div(ctx, tmp_out[i], den).cell
        })
        .collect();
    out.floats(&result);
    Ok(())
}
