//! DS-1000 case 0309: literal translation of kernel.py (LITERAL_BASELINES.md).
//! Linear scan, strict comparison keeps the first occurrence; fresh bounds per comparison.
//! Unravel uses integer quotient/remainder of the selected index.
use crate::prelude::*;
pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[8, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);
    let a = q.load_floats(ctx, &a);
    let mut best = a[0];
    let mut idx = q.int_const(ctx, 0);
    for i in 1..48 {
        let x = q.bound(ctx, a[i]);
        let b = q.bound(ctx, best);
        let better = q.gt(ctx, x, b);
        best = q.gate.select(ctx, a[i], best, better);
        idx = q.gate.select(ctx, Constant(fe(i as i128)), idx, better);
    }
    out.int(idx);
    Ok(())
}
