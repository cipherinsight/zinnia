//! DS-1000 case 0437, halo2-base baseline.
//! Kernel: `a.min(axis=1, keepdims=1) == a` on `Float[10, 5]` (exact equality).
//! A row minimum is `q.min_of` (a hint pinned by five range checks and a
//! product).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[10, 5])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let mut mask = Vec::with_capacity(50);
    for row in a.chunks(5) {
        let min = q.min_of(ctx, row);
        for x in row {
            mask.push(q.eq(ctx, *x, min));
        }
    }
    out.bools(&mask);
    Ok(())
}
