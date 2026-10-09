//! npbench_deep_learning_softmax_softmax__softmax (NPBench deep_learning/softmax), halo2-base baseline.
//! Kernel: softmax over the last axis of x: Float[N, H, SM, SM]:
//! `tmp_max = np.max(x, axis=-1, keepdims=True)`,
//! `tmp_out = np.exp(x - tmp_max)`, `tmp_sum = np.sum(tmp_out, axis=-1,
//! keepdims=True)`, `tmp_out / tmp_sum`.
//! Each row's maximum is `q.max_of` (a hinted maximum, every gap `max - x`
//! range-checked to `[0, 2^64)`; 542,528 cells against 564,032 for a chain of
//! comparisons), and those checks are the precondition of `q.exp_neg` (the
//! shared gadget, NPB_BASELINES.md section 5, rule 5), so `exp(x - max)` is
//! `exp_neg(max - x)`. The sum is exact; each quotient is a float division by
//! the sum, a positive integer below 2^(33 + bits(SM)) (each term is in
//! `[0, 2^33)` and the row's maximum contributes `exp(0)`), so `div_by_abs`
//! takes it as its own magnitude.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 2; // = x_0
const H: usize = 2; // = x_1
const SN: usize = 16; // = x_2
const SM: usize = 16; // = x_3

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.floats("x", &[N, H, SN, SM])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_floats(ctx, &x);
    for row in x.chunks(SM) {
        let largest = q.max_of(ctx, row);
        let gaps: Vec<Cell> = row.iter().map(|v| q.sub(ctx, largest, *v)).collect();
        let exps: Vec<Cell> = gaps.iter().map(|g| q.exp_neg(ctx, *g).cell).collect();
        let total = q.sum(ctx, exps.iter().copied());
        let quotients: Vec<Cell> = exps.iter().map(|e| q.div_by_abs(ctx, *e, total, total).cell).collect();
        out.floats(&quotients);
    }
    Ok(())
}
