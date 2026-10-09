//! DS-1000 case 0491: literal translation of kernel.py (LITERAL_BASELINES.md).
//! isclose uses abs(a-b) <= atol + rtol*abs(b); defaults quantize to 43 and 42950. Full boolean arrays precede sum.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let ra = inputs.floats("s1", &[6, 1])?;
    let rb = inputs.floats("s2", &[6, 1])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &ra);
    let b = q.load_floats(ctx, &rb);
    let mut close = Vec::new();
    for i in 0..6 {
        let diff = q.sub(ctx, a[i], b[i]);
        let d = q.abs_int64(ctx, diff);
        let ab = q.abs_int64(ctx, b[i]);
        let relative = q.mul(ctx, Constant(fe(quantize(1e-5))), ab).cell;
        let tol = q.add(ctx, Constant(fe(quantize(1e-8))), relative);
        let lhs = q.bound(ctx, d);
        let rhs = q.bound(ctx, tol);
        close.push(q.le(ctx, lhs, rhs));
    }
    let not_close: Vec<Cell> = close.iter().map(|c| q.gate.not(ctx, *c)).collect();
    let mut result = ctx.load_zero();
    for v in not_close {
        result = q.add(ctx, result, v);
    }
    out.int(result);
    Ok(())
}
