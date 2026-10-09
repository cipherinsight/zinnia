//! pythran_cronbach__cronbach (Pythran cronbach), halo2-base baseline.
//! Kernel: Cronbach's alpha of `itemscores: Float[ROWS, COLS]`:
//! `ROWS / (ROWS - 1) * (1 - itemvars.sum() / tscores.var(ddof=1))` with
//! `itemvars` the rows' variances (`ddof=1`) and `tscores` the column sums.
//! A variance is NumPy's `_var`: the mean `sum / n` (a quotient by the
//! integer constant `n`, `q.floor_div_const`), the sum of squared deviations
//! (one reduction, rescaled once: NPB_BASELINES.md section 5.3), divided by
//! `n - 1`. `ROWS / (ROWS - 1)` is a float64 constant of the kernel, so the
//! last step is one rounded product with its Q32 code. The total variance is
//! a sum of squares, so the division by it is `q.div_pos` (a zero variance
//! has no witness, as NumPy's result is not finite).

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const ROWS: usize = 16; // = itemscores_0
const COLS: usize = 16; // = itemscores_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let itemscores = inputs.floats("itemscores", &[ROWS, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let scores = q.load_floats(ctx, &itemscores);
    // itemvars = itemscores.var(1, None, None, 1)
    let itemvars: Vec<Cell> = scores.chunks(COLS).map(|row| var_ddof1(&q, ctx, row).cell).collect();
    // tscores = itemscores.sum(0)
    let tscores: Vec<Cell> =
        (0..COLS).map(|j| q.sum(ctx, (0..ROWS).map(|i| scores[i * COLS + j]))).collect();
    let tvar = var_ddof1(&q, ctx, &tscores);
    let itemvars_sum = q.sum(ctx, itemvars);
    let ratio = q.div_pos(ctx, itemvars_sum, tvar);
    let one_minus = q.sub(ctx, Constant(fe(SCALE)), ratio.cell);
    let factor = Constant(fe(quantize(ROWS as f64 / (ROWS - 1) as f64)));
    let alpha = q.mul(ctx, factor, one_minus);
    out.float(alpha.cell);
    Ok(())
}

/// `xs.var(ddof=1)`: `sum((x - mean)^2) / (n - 1)` with `mean = sum(x) / n`.
fn var_ddof1(q: &Q32, ctx: &mut Ctx, xs: &[Cell]) -> I64 {
    let n = xs.len() as u64;
    let total = q.sum(ctx, xs.iter().copied());
    let (mean, _) = q.floor_div_const(ctx, total, n);
    let deviations: Vec<Cell> = xs.iter().map(|x| q.sub(ctx, *x, mean.cell)).collect();
    let squares = q.dot(ctx, &deviations, &deviations);
    q.floor_div_const(ctx, squares.cell, n - 1).0
}
