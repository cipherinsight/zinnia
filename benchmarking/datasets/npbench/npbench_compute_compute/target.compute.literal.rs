//! npbench_compute_compute__compute (NPBench compute), halo2-base literal baseline.
//! LITERAL_BASELINES.md: direct translation of kernel.py, no optimization.
//!
//! Source (one statement, kernel.py):
//!   return np.clip(array_1, 2, 10) * a + array_2 * b + c
//! Translation choices:
//! - `np.clip` on an int64 array (L5 table): `q.clip_const`, the shared
//!   region-pinning gadget for a clip against compile-time constant bounds.
//! - `array_1, array_2, a, b, c` are `Integer` (not `Number`): plain int64
//!   arithmetic, so `*` and `+` use the raw field ops `q.gate.mul`/`q.gate.add`
//!   (no Q32 rescale; `q.mul` would wrongly divide by 2^32).
//! - The single expression is evaluated in the same left-to-right grouping
//!   Python uses: `((clip(array_1) * a) + (array_2 * b)) + c`, one gate call
//!   per operator, no fusing of the two `*` with the trailing `+`.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = array_1_0
const COLS: usize = 16; // = array_1_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let array_1 = inputs.ints("array_1", &[N, COLS])?;
    let array_2 = inputs.ints("array_2", &[N, COLS])?;
    let (a, b, c) = (inputs.int("a")?, inputs.int("b")?, inputs.int("c")?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let array_1 = q.load_ints_bounded(ctx, &array_1);
    let array_2 = q.load_ints(ctx, &array_2);
    let a = q.load_int(ctx, a);
    let b = q.load_int(ctx, b);
    let c = q.load_int(ctx, c);

    let result: Vec<Cell> = array_1
        .iter()
        .zip(&array_2)
        .map(|(x, y)| {
            let clipped = q.clip_const(ctx, *x, 2, 10);
            let term1 = q.gate.mul(ctx, clipped, a);
            let term2 = q.gate.mul(ctx, *y, b);
            let sum = q.gate.add(ctx, term1, term2);
            q.gate.add(ctx, sum, c)
        })
        .collect();
    out.ints(&result);
    Ok(())
}
