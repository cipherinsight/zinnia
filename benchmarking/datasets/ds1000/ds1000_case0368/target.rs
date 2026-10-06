//! DS-1000 case 0368, halo2-base baseline.
//! Kernel: `np.isclose(a, a[0], atol=0).all()` on `Float[3, 3]`: every row
//! close to row 0 by NumPy's `|x - y| <= atol + rtol * |y|` with `rtol = 1e-5`,
//! `atol = 0`, `y = a[0, j]`.
//! Row 0 against itself is always close (`0 <= rtol * |y|`) and is skipped.
//! The tolerance `rtol * |a[0, j]|` (a rescaled product) is computed once per
//! column; each of the 6 other values costs one magnitude `|x - y|` (one range
//! check) and one comparison. `.all()` is "no value is far".

use crate::prelude::*;

const N: usize = 3;
const RTOL: f64 = 1e-5;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[N, N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let tol: Vec<Cell> = (0..N).map(|j| tolerance(&q, ctx, a[j])).collect();
    let mut far = Vec::with_capacity((N - 1) * N);
    for i in 1..N {
        for j in 0..N {
            far.push(is_far(&q, ctx, a[i * N + j], a[j], tol[j]));
        }
    }
    let n_far = q.sum(ctx, far);
    let all_close = q.gate.is_zero(ctx, n_far);
    out.bool(all_close);
    Ok(())
}

/// `rtol * |y|` as a rescaled product; it lies in `[0, 2^63)` because `|y|`
/// does and the rescale is the floor of a non-negative product. `|y|` is
/// `q.abs_int64`: `y = -2^31` (Q32 code `i64::MIN`) has no proof, as in Zinnia.
fn tolerance(q: &Q32, ctx: &mut Ctx, y: Cell) -> Cell {
    let abs_y = q.abs_int64(ctx, y);
    q.mul(ctx, abs_y, Constant(fe(quantize(RTOL)))).cell
}

/// `!isclose(x, y)`, i.e. `tol < |x - y|`. `q.abs_int64` proves
/// `|x - y| < 2^63` (a difference of Q32 code `-2^63` or outside the window has
/// no proof, as in Zinnia), so both sides lie in `[0, 2^64)`, as halo2-base's
/// 64-bit `is_less_than` requires.
fn is_far(q: &Q32, ctx: &mut Ctx, x: Cell, y: Cell, tol: Cell) -> Cell {
    let diff = q.sub(ctx, x, y);
    let gap = q.abs_int64(ctx, diff);
    q.range.is_less_than(ctx, tol, gap, 64)
}
