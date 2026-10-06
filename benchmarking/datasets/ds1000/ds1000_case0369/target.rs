//! DS-1000 case 0369, halo2-base baseline.
//! Kernel: `np.isclose(a, a[:, 0].reshape(-1, 1), atol=0).all()` on
//! `Float[3, 3]`: every column close to column 0 by NumPy's
//! `|x - y| <= atol + rtol * |y|` with `rtol = 1e-5`, `atol = 0`, `y = a[i, 0]`.
//! Column 0 against itself is always close (`0 <= rtol * |y|`) and is skipped.
//! The tolerance `rtol * |a[i, 0]|` (a rescaled product) is computed once per
//! row; each of the 6 other values costs one window bound of `x - y`, its sign
//! test, and one comparison. `.all()` is "no value is far".

use crate::prelude::*;

const N: usize = 3;
const RTOL: f64 = 1e-5;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[N, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let ab = q.load_floats_bounded(ctx, &a);
    let a: Vec<Cell> = ab.iter().map(|x| x.cell).collect();
    let mut far = Vec::with_capacity(N * (N - 1));
    for i in 0..N {
        let y = a[i * N];
        let tol = tolerance(&q, ctx, ab[i * N]);
        for j in 1..N {
            far.push(is_far(&q, ctx, a[i * N + j], y, tol));
        }
    }
    let n_far = q.sum(ctx, far);
    let all_close = q.gate.is_zero(ctx, n_far);
    out.bool(all_close);
    Ok(())
}

/// `rtol * |y|` as a rescaled product; it lies in `[0, 2^63)` because `|y|`
/// does and the rescale is the floor of a non-negative product.
fn tolerance(q: &Q32, ctx: &mut Ctx, y: I64) -> Cell {
    let abs_y = q.abs(ctx, y);
    q.mul(ctx, abs_y, Constant(fe(quantize(RTOL)))).cell
}

/// `!isclose(x, y)`, i.e. `tol < |x - y|`. Both sides lie in `[0, 2^64)`
/// (`|x - y| <= 2^63` once `x - y` is in the window), as halo2-base's 64-bit
/// `is_less_than` requires.
fn is_far(q: &Q32, ctx: &mut Ctx, x: Cell, y: Cell, tol: Cell) -> Cell {
    let diff = q.sub(ctx, x, y);
    let diff = q.bound(ctx, diff);
    let gap = q.abs(ctx, diff);
    q.range.is_less_than(ctx, tol, gap, 64)
}
