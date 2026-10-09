//! npbench_polybench_trisolv_trisolv__kernel (NPBench polybench/trisolv),
//! halo2-base literal baseline. LITERAL_BASELINES.md: direct translation of
//! kernel.py, no optimization.
//!
//! Source (kernel.py):
//!   for i in range(x.shape[0]):
//!       x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]
//! Translation choices:
//! - The loop runs in source order, one circuit statement per iteration; `x`
//!   is overwritten before any later statement reads it and the strict upper
//!   triangle of `L` is never read by any iteration, so neither is loaded
//!   (CONVENTIONS 1.6: load/check only what is read).
//! - `L[i, :i] @ x[:i]` (L5 table, single `@` call): one reduction, rescaled
//!   once (`q.dot`), matching how NumPy rounds `@` once. `np.dot([], [])` is
//!   `0.0`; `q.dot` on an empty pair would need a non-empty inner product, so
//!   `i = 0` is special-cased to the constant `0` it always evaluates to,
//!   not an algorithmic shortcut.
//! - `/ L[i, i]`: a single plain division of the subtraction's result by the
//!   diagonal element. L6 forbids `q.div_by_abs` (a divisor magnitude
//!   reused from elsewhere); here the divisor's magnitude is used nowhere
//!   else, so the direct general-division gadget `q.div` is the literal
//!   choice. `q.div` takes the divisor's sign and magnitude from its own
//!   window proof (`q.abs` internally), which is what source `abs()` would
//!   do for a plain division; this also correctly accepts the divisor
//!   `-2^63`, which is a valid Q32 code in the window.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = L_0

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let l = inputs.floats("L", &[N, N])?;
    let b = inputs.floats("b", &[N])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let b = q.load_floats(ctx, &b);
    let mut x: Vec<Cell> = Vec::with_capacity(N);
    for i in 0..N {
        let row = q.load_floats(ctx, &l[i * N..i * N + i]);
        let diagonal = q.load_float_bounded(ctx, l[i * N + i]);
        let rest = if i == 0 {
            b[0]
        } else {
            let dot = q.dot(ctx, &row, &x);
            q.sub(ctx, b[i], dot.cell)
        };
        x.push(q.div(ctx, rest, diagonal).cell);
    }
    out.floats(&x);
    Ok(())
}
