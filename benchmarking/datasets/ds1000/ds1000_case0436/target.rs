//! DS-1000 case 0436, halo2-base baseline.
//! Kernel: `a.max(axis=1, keepdims=1) == a` on `Float[10, 5]` (exact equality).
//! Each row maximum is `q.max_of` (a hint pinned by five range checks and a
//! product); the mask compares every value with its row maximum.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[10, 5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let mut mask = Vec::with_capacity(50);
    for row in a.chunks(5) {
        let max = q.max_of(ctx, row);
        for x in row {
            mask.push(q.eq(ctx, *x, max));
        }
    }
    out.bools(&mask);
    Ok(())
}
