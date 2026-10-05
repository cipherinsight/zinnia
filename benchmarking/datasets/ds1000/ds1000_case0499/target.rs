//! DS-1000 case 0499, halo2-base baseline.
//! Kernel: `to_shape(a, (93, 13))` on `Float[42, 10]` (`shape` is a constant):
//! `np.pad` centring `a`, with the odd leftover on the bottom/right:
//! 51 padding rows split 25 above and 26 below, 3 padding columns split 1 left
//! and 2 right. The `a` cells are exposed as they are; the zeros are circuit
//! constants, not public cells, declared between the rows.

use crate::prelude::*;

const TOP: usize = 25;
const BOTTOM: usize = 26;
const LEFT: usize = 1;
const RIGHT: usize = 2;
const WIDTH: usize = 13;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[42, 10])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    out.const_floats(&vec![0.0; TOP * WIDTH + LEFT]);
    for (i, row) in a.chunks(10).enumerate() {
        out.floats(row);
        // The row's right padding, then the next row's left padding or, after
        // the last row, the bottom padding rows.
        let zeros = if i < 41 { RIGHT + LEFT } else { RIGHT + BOTTOM * WIDTH };
        out.const_floats(&vec![0.0; zeros]);
    }
    Ok(())
}
