//! DS-1000 case 0496, halo2-base baseline.
//! Kernel: `np.pad(a, ((0, 93 - 42), (0, 13 - 10)), 'constant')` on
//! `Float[42, 10]` (`shape = (93, 13)` is a constant): each row of `a` followed
//! by 3 zeros, then 51 zero rows. The `a` cells are exposed as they are; the
//! zeros are circuit constants, not public cells, declared between the rows.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[42, 10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    for (i, row) in a.chunks(10).enumerate() {
        out.floats(row);
        // The row's 3 padding columns; after the last row also the 51 zero rows.
        let zeros = if i < 41 { 3 } else { 3 + 51 * 13 };
        out.const_floats(&vec![0.0; zeros]);
    }
    Ok(())
}
