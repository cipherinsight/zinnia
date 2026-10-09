// pythran_fibo_seq__fibo: literal translation of kernel.py (LITERAL_BASELINES.md).
// Private range(n) is bounded by the explicit 1024 precondition with selected state. Result is int64 checked.
use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let n = inputs.int("n")?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let n = q.load_int(ctx, n);
    let nb = q.bound(ctx, n);
    let lo = q.bounded_const(ctx, 0);
    let lower = q.le(ctx, lo, nb);
    q.gate.assert_is_const(ctx, &lower, &Fr::ONE);
    let nb = q.bound(ctx, n);
    let hi = q.bounded_const(ctx, 1024);
    let upper = q.le(ctx, nb, hi);
    q.gate.assert_is_const(ctx, &upper, &Fr::ONE);
    let mut a = q.int_const(ctx, 1);
    let mut b = q.int_const(ctx, 1);
    for step in 0..1024 {
        let nb = q.bound(ctx, n);
        let counter = q.bounded_const(ctx, step as i128);
        let active = q.lt(ctx, counter, nb);
        let next = q.add(ctx, a, b);
        let na = q.gate.select(ctx, next, a, active);
        b = q.gate.select(ctx, a, b, active);
        a = na;
    }
    // Python integer output must fit the statement's int64 window.
    q.bound(ctx, a);
    out.int(a);
    Ok(())
}
