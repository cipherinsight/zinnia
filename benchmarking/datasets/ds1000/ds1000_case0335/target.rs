//! DS-1000 case 0335, halo2-base baseline.
//! Kernel: `np.max([a, b, c], axis=0)` on three `Float[50]`: the elementwise maximum.
//! Each maximum is `q.max_of`: a hint pinned by three range checks and a product
//! (instead of three window bounds and two comparisons of a compare-and-select
//! chain).

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let (a, b, c) = (inputs.floats("a", &[50])?, inputs.floats("b", &[50])?, inputs.floats("c", &[50])?);
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let (a, b, c) = (q.load_floats(ctx, &a), q.load_floats(ctx, &b), q.load_floats(ctx, &c));
    let max: Vec<Cell> = (0..50).map(|i| q.max_of(ctx, &[a[i], b[i], c[i]])).collect();
    out.floats(&max);
    Ok(())
}

