//! Sequential Fibonacci recurrence with a private trip count, as kernel.py.
//! The source bound is 1024; no matrix powering or derived n<=90 shortcut.
//! State arithmetic is exact field addition, matching the frozen compiler.
//! The intended int64-result comparison covers n<=90; no extra range guards.
use crate::prelude::*;
const MAX_STEPS: usize = 1024;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let n = inputs.int("n")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let n = q.load_int_bounded(ctx, n);
    let negative = q.lt_const(ctx, n, 0);
    q.gate.assert_is_const(ctx, &negative, &Fr::ZERO);
    let in_range = q.lt_const(ctx, n, (MAX_STEPS + 1) as i128);
    q.gate.assert_is_const(ctx, &in_range, &Fr::ONE);
    let mut a = q.int_const(ctx, 1);
    let mut b = a;
    for step in 0..MAX_STEPS {
        let active = q.gt_const(ctx, n, step as i128);
        let next = q.add(ctx, a, b);
        let next_a = q.gate.select(ctx, next, a, active);
        b = q.gate.select(ctx, a, b, active);
        a = next_a;
    }
    out.int(a);
    Ok(())
}
