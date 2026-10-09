//! npbench_compute_compute__compute (NPBench compute), halo2-base baseline.
//! Kernel: `np.clip(array_1, 2, 10) * a + array_2 * b + c` on int64 arrays
//! `array_1, array_2: Integer[N, COLS]` and int64 scalars `a, b, c`.
//! The clip is `q.clip_const` (region bits pinned by one 64-bit range check)
//! on the window proof of each `array_1` element's input check; the rest is
//! exact integer arithmetic.

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
            let partial = q.gate.mul_add(ctx, clipped, a, c);
            q.gate.mul_add(ctx, *y, b, partial)
        })
        .collect();
    out.ints(&result);
    Ok(())
}
