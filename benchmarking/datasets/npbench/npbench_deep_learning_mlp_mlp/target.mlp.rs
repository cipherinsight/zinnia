//! npbench_deep_learning_mlp_mlp__mlp (NPBench deep_learning/mlp), halo2-base baseline.
//! Kernel: a three-layer perceptron on input: Float[N, C_IN]:
//! `x = relu(input @ w1 + b1)`, `x = relu(x @ w2 + b2)`,
//! `softmax(x @ w3 + b3)` over the last axis, with `relu(x) = np.maximum(x, 0)`
//! and softmax as in the softmax entry (`exp(x - max) / sum`).
//! Each element of a matrix product is one reduction, rescaled once; adding
//! the bias after the rescale is exact, so the bias joins the rescaled sum
//! (`floor((sum + b 2^32) / 2^32) = floor(sum / 2^32) + b`), whose quotient is
//! window-checked by the rescale and feeds relu's sign test directly.
//! Softmax: each row's maximum is `q.max_of`, whose gap checks are the
//! precondition of `q.exp_neg` (the shared gadget, NPB_BASELINES.md section 5,
//! rule 5); the sum is exact and each quotient divides by it, a positive
//! integer below 2^(33 + bits(S2)) and its own magnitude (`div_by_abs`).

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

    let x = dense(&q, ctx, &input, &w1, &b1, C_IN, S0);
    let x: Vec<Cell> = x.iter().map(|v| relu(&q, ctx, *v)).collect();
    let x = dense(&q, ctx, &x, &w2, &b2, S0, S1);
    let x: Vec<Cell> = x.iter().map(|v| relu(&q, ctx, *v)).collect();
    let x = dense(&q, ctx, &x, &w3, &b3, S1, S2);
    for row in x.chunks(S2) {
        let row: Vec<Cell> = row.iter().map(|v| v.cell).collect();
        let largest = q.max_of(ctx, &row);
        let exps: Vec<Cell> = row
            .iter()
            .map(|v| {
                let gap = q.sub(ctx, largest, *v);
                q.exp_neg(ctx, gap).cell
            })
            .collect();
        let total = q.sum(ctx, exps.iter().copied());
        let quotients: Vec<Cell> = exps.iter().map(|e| q.div_by_abs(ctx, *e, total, total).cell).collect();
        out.floats(&quotients);
    }
    Ok(())
}

/// `x @ w + b` for `x: [rows, inner]`, `w: [inner, cols]`, `b: [cols]`: each
/// element `floor((sum_k x w + b 2^32) / 2^32)`, window-checked.
fn dense(q: &Q32, ctx: &mut Ctx, x: &[Cell], w: &[Cell], b: &[Cell], inner: usize, cols: usize) -> Vec<I64> {
    x.chunks(inner)
        .flat_map(|row| {
            (0..cols)
                .map(|c| {
                    let column = (0..inner).map(|k| Existing(w[k * cols + c]));
                    let sum = q.gate.inner_product(ctx, row.iter().map(|v| Existing(*v)), column);
                    let biased = q.gate.mul_add(ctx, b[c], Constant(fe(SCALE)), sum);
                    q.rescale(ctx, biased)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `np.maximum(x, 0)`.
fn relu(q: &Q32, ctx: &mut Ctx, x: I64) -> Cell {
    let negative = q.is_neg(ctx, x);
    q.gate.select(ctx, Constant(Fr::ZERO), x.cell, negative)
}
