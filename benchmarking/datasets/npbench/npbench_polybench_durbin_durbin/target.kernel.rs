//! npbench_polybench_durbin_durbin__kernel (NPBench polybench/durbin), halo2-base baseline.
//! Kernel: Durbin's recursion on `r: Float[N]`: `y[0] = alpha = -r[0]`,
//! `beta = 1.0`; for `k` in `1..N`: `beta *= 1.0 - alpha * alpha`,
//! `alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta`,
//! `y[:k] += alpha * np.flip(y[:k])`, `y[k] = alpha`; returns `y`.
//! Rounding in NumPy's order: each product and the quotient is rounded, the
//! dot product once (`q.dot`), the negation applies before the division. The
//! first `beta *= ...` multiplies by the constant 1.0, which is exact, so it
//! is skipped; `alpha * np.flip(y[:k])` reads `y` before the update.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = r_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let r = inputs.floats("r", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let r = q.load_floats(ctx, &r);
    let mut y: Vec<Cell> = Vec::with_capacity(N);
    let mut alpha = q.neg(ctx, r[0]);
    y.push(alpha);
    let mut beta: Option<I64> = None; // None: the initial 1.0
    for k in 1..N {
        let square = q.mul(ctx, alpha, alpha);
        let factor = q.sub(ctx, Constant(fe(SCALE)), square.cell);
        beta = Some(match beta {
            None => q.bound(ctx, factor),
            Some(b) => q.mul(ctx, b, factor),
        });
        let reflected: Vec<Cell> = r[..k].iter().rev().copied().collect();
        let dot = q.dot(ctx, &reflected, &y);
        let sum = q.add(ctx, r[k], dot.cell);
        let numerator = q.neg(ctx, sum);
        alpha = q.div(ctx, numerator, beta.expect("set above")).cell;
        let flipped: Vec<Cell> = (0..k).map(|j| q.mul(ctx, alpha, y[k - 1 - j]).cell).collect();
        for (yj, f) in y.iter_mut().zip(flipped) {
            *yj = q.add(ctx, *yj, f);
        }
        y.push(alpha);
    }
    out.floats(&y);
    Ok(())
}
