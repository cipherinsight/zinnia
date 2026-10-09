//! npbench_go_fast_go_fast__go_fast (NPBench go_fast), halo2-base baseline.
//! Kernel: on a: Float[N, COLS], `trace = sum_i np.tanh(a[i, i])` (from
//! 0.0, over the diagonal), then `a + trace`. `q.tanh` is the shared gadget
//! (NPB_BASELINES.md section 5, rule 5); the sums are exact.

use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = a_0
const COLS: usize = 16; // = a_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats_bounded(ctx, &a);
    // The kernel's loop runs over a.shape[0]; a[i, i] needs i < COLS.
    if N > COLS {
        return Err(Error::new("a[i, i] is out of bounds: NumPy raises IndexError"));
    }
    let tanhs: Vec<Cell> = (0..N).map(|i| q.tanh(ctx, a[i * COLS + i]).cell).collect();
    let trace = q.sum(ctx, tanhs);
    let result: Vec<Cell> = a.iter().map(|x| q.add(ctx, x.cell, trace)).collect();
    out.floats(&result);
    Ok(())
}
