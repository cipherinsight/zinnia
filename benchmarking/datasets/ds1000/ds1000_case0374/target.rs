//! DS-1000 case 0374, halo2-base baseline.
//! Kernel: ECDF lookup on `grades: Float[50]`, `eval: Integer[5]`: with
//! `xs = sort(grades)` and `ys = arange(1, 51) / 50`, each `e` of `eval` maps to
//! `0` if `e < xs[0]`, `1` if `e >= xs[-1]`, else `ys[(xs > e).argmax() - 1]`.
//! For sorted `xs`, `(xs > e).argmax()` is the number `c` of grades `<= e`
//! whenever some grade exceeds `e`, so every branch equals `c / 50` (`c = 0`
//! below `xs[0]`, `c = 50` from `xs[-1]` on), ties included. The circuit counts
//! `c` directly (50 comparisons per element, no sort) and returns the constant
//! quotient `floor((c * 2^32 + 25) / 50) = round(c / 50 * 2^32)`, the kernel's
//! Q32 constant `ys[c - 1] = c / 50` (rounded, as Zinnia and NumPy quantize it).
//! `e` is compared as the float `e * 2^32`, which must lie in the window
//! (`-2^31 <= e < 2^31`).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (grades, eval) = (inputs.floats("grades", &[50])?, inputs.ints("eval", &[5])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let grades = q.load_floats_bounded(ctx, &grades);
    let eval = q.load_ints(ctx, &eval);
    let mut result = Vec::with_capacity(eval.len());
    for e in eval {
        let e = q.int_to_float(ctx, e);
        let e = q.bound(ctx, e);
        // c = 50 - #{g : g > e}; c * 2^32 + 25 rounds the quotient by 50
        let above: Vec<Cell> = grades.iter().map(|g| q.lt(ctx, e, *g)).collect();
        let n_above = q.sum(ctx, above);
        let c_scaled = q.gate.mul_add(ctx, n_above, Constant(fe(-SCALE)), Constant(fe(50 * SCALE + 25)));
        result.push(q.floor_div_const(ctx, c_scaled, 50).0.cell);
    }
    out.floats(&result);
    Ok(())
}
