// npbench_go_fast_go_fast__go_fast: literal translation of kernel.py (LITERAL_BASELINES.md).
// CHECKER BLOCKER: Halo2 L5/L6 per-operation input bounds conflict with the checker's
// zero redundant_input_checks requirement; see LITERAL_NPB3.md. No diagnostic bypass.
// Sequential trace recurrence; canonical shared tanh matches the optimized arithmetic.
use crate::prelude::*;

// Sizes (hw_check.sizes): from fixture.json.
const N: usize = 16; // = a_0
const COLS: usize = 16; // = a_1

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[N, COLS])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let mut trace = q.float_const(ctx, 0.0);
    for i in 0..N {
        let v = q.bound(ctx, a[i * COLS + i]);
        let t = q.canonical_tanh(ctx, v).cell;
        trace = q.add(ctx, trace, t);
    }
    let result: Vec<Cell> = a.iter().map(|x| q.add(ctx, *x, trace)).collect();
    out.floats(&result);
    Ok(())
}
