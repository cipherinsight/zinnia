// npbench_deep_learning_mlp_mlp__mlp: literal translation of kernel.py (LITERAL_BASELINES.md).
// Matmuls rescale and window-check each dot (as the optimized baseline does since b6fb8bf3), then the source adds the bias separately. ReLU/softmax are source helpers; max and sum scan in index order, canonical exp per element.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 2; // = input_0
const C_IN: usize = 2; // = input_1
const S0: usize = 16; // = w1_1
const S1: usize = 2; // = w2_1
const S2: usize = 2; // = w3_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let input = inputs.floats("input", &[N, C_IN])?;
    let w1 = inputs.floats("w1", &[C_IN, S0])?;
    let b1 = inputs.floats("b1", &[S0])?;
    let w2 = inputs.floats("w2", &[S0, S1])?;
    let b2 = inputs.floats("b2", &[S1])?;
    let w3 = inputs.floats("w3", &[S1, S2])?;
    let b3 = inputs.floats("b3", &[S2])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let input = q.load_floats(ctx, &input);
    let w1 = q.load_floats(ctx, &w1);
    let b1 = q.load_floats(ctx, &b1);
    let w2 = q.load_floats(ctx, &w2);
    let b2 = q.load_floats(ctx, &b2);
    let w3 = q.load_floats(ctx, &w3);
    let b3 = q.load_floats(ctx, &b3);
    let product = matmul(&q, ctx, &input, &w1, C_IN, S0);
    let biased: Vec<Cell> = (0..N * S0)
        .map(|t| q.add(ctx, product[t], b1[t % S0]))
        .collect();
    let x = relu(&q, ctx, &biased);
    let product = matmul(&q, ctx, &x, &w2, S0, S1);
    let biased: Vec<Cell> = (0..N * S1)
        .map(|t| q.add(ctx, product[t], b2[t % S1]))
        .collect();
    let x = relu(&q, ctx, &biased);
    let product = matmul(&q, ctx, &x, &w3, S1, S2);
    let biased: Vec<Cell> = (0..N * S2)
        .map(|t| q.add(ctx, product[t], b3[t % S2]))
        .collect();
    let x = softmax(&q, ctx, &biased);
    out.floats(&x);
    Ok(())
}
// L5 defining matrix product, one rescale per output.
fn matmul(q: &Q32, ctx: &mut Ctx, x: &[Cell], w: &[Cell], inner: usize, cols: usize) -> Vec<Cell> {
    let mut result = Vec::with_capacity(N * cols);
    for r in 0..N {
        for c in 0..cols {
            let mut raw = q.int_const(ctx, 0);
            for k in 0..inner {
                let term = q.gate.mul(ctx, x[r * inner + k], w[k * cols + c]);
                raw = q.add(ctx, raw, term);
            }
            result.push(q.rescale(ctx, raw).cell);
        }
    }
    result
}
// Python relu helper: np.maximum(x,0).
fn relu(q: &Q32, ctx: &mut Ctx, x: &[Cell]) -> Vec<Cell> {
    x.iter()
        .map(|v| {
            let a = q.bound(ctx, *v);
            let b = q.bounded_const(ctx, 0);
            q.max(ctx, a, b).cell
        })
        .collect()
}
// Python softmax helper, with complete arrays at each statement.
fn softmax(q: &Q32, ctx: &mut Ctx, x: &[Cell]) -> Vec<Cell> {
    let mut tmp_max = Vec::with_capacity(N);
    for r in 0..N {
        let mut best = x[r * S2];
        for c in 1..S2 {
            let a = q.bound(ctx, best);
            let b = q.bound(ctx, x[r * S2 + c]);
            best = q.max(ctx, a, b).cell;
        }
        tmp_max.push(best);
    }
    let difference: Vec<Cell> = (0..N * S2)
        .map(|t| q.sub(ctx, x[t], tmp_max[t / S2]))
        .collect();
    let tmp_out: Vec<Cell> = difference
        .iter()
        .map(|v| {
            let x = q.bound(ctx, *v);
            q.canonical_exp(ctx, x).cell
        })
        .collect();
    let mut tmp_sum = Vec::with_capacity(N);
    for r in 0..N {
        let mut acc = q.float_const(ctx, 0.0);
        for c in 0..S2 {
            acc = q.add(ctx, acc, tmp_out[r * S2 + c]);
        }
        tmp_sum.push(acc);
    }
    (0..N * S2)
        .map(|t| {
            let d = q.bound(ctx, tmp_sum[t / S2]);
            q.div(ctx, tmp_out[t], d).cell
        })
        .collect()
}
