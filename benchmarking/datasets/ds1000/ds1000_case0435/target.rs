//! DS-1000 case 0435, halo2-base baseline.
//! Kernel: `a[1, :] = 0; a[:, 0] = 0; return a` on `Integer[4, 4]`.
//! The zeroed cells (row 1 and column 0) are circuit constants, not public
//! cells; only the 9 cells that keep their input value are loaded and exposed,
//! in row-major order between the constant segments.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.ints("a", &[4, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let row0 = q.load_ints(ctx, &a[1..4]);
    let row2 = q.load_ints(ctx, &a[9..12]);
    let row3 = q.load_ints(ctx, &a[13..16]);
    out.const_ints(&[0]); // a[0, 0]
    out.ints(&row0); // a[0, 1:]
    out.const_ints(&[0; 5]); // a[1, :], a[2, 0]
    out.ints(&row2); // a[2, 1:]
    out.const_ints(&[0]); // a[3, 0]
    out.ints(&row3); // a[3, 1:]
    Ok(())
}
