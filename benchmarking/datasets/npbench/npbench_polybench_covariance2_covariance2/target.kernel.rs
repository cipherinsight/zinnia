//! npbench_polybench_covariance2_covariance2__kernel (NPBench polybench/covariance2), halo2-base baseline.
//! Kernel: `np.cov(np.transpose(data))` on data: Float[NO, NV] (NV variables,
//! NO observations), with the precondition `0 <= M <= 1024` on M: Integer;
//! float_n is never read and not loaded.
//! NumPy's `cov`: the mean of each variable (`sum / NO`, a float division by
//! the integer count: `floor(sum / NO)` in Q32), the deviations (exact), the
//! product of the deviation matrix with its transpose (each element one
//! reduction, rescaled once) and `c *= 1 / (NO - 1)` (a rounded product with
//! the constant `1 / (NO - 1)`, as NumPy multiplies by the reciprocal). The
//! product is symmetric, so each pair is computed once and output twice.
//! M is read only by the precondition (NPB_BASELINES.md section 5, rule 2):
//! it is proven to lie in [0, 1024], which implies its domain (an int64).

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
    let m = ctx.load_witness(fe(m as i128));
    q.range.range_check(ctx, m, 16);
    let headroom = q.sub(ctx, Constant(fe(1024)), m);
    q.range.range_check(ctx, headroom, 16);

    let data = q.load_floats(ctx, &data);
    let deviations: Vec<Vec<Cell>> = (0..NV)
        .map(|v| {
            let column: Vec<Cell> = (0..NO).map(|o| data[o * NV + v]).collect();
            let total = q.sum(ctx, column.iter().copied());
            let (mean, _) = q.floor_div_const(ctx, total, NO as u64);
            column.iter().map(|x| q.sub(ctx, *x, mean.cell)).collect()
        })
        .collect();
    let reciprocal = Constant(fe(quantize(1.0 / (NO - 1) as f64)));
    let mut cov: Vec<Option<Cell>> = vec![None; NV * NV];
    for a in 0..NV {
        for b in a..NV {
            let product = q.dot(ctx, &deviations[a], &deviations[b]);
            let c = q.mul(ctx, product.cell, reciprocal).cell;
            cov[a * NV + b] = Some(c);
            cov[b * NV + a] = Some(c);
        }
    }
    let cov: Vec<Cell> = cov.into_iter().map(|c| c.expect("every pair")).collect();
    out.floats(&cov);
    Ok(())
}
