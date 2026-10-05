//! DS-1000 case 0389, halo2-base baseline.
//! Kernel: `a.reshape(6 // 3, -1, 3, 4).swapaxes(1, 2).reshape(6, 12)` on
//! `Float[6, 3, 4]` (`h = 6`, `w = 12` are constants): the six 3x4 blocks laid
//! out as a 2 x 3 grid of blocks, so output (R, C) is
//! `a[(R / 3) * 3 + C / 4, R % 3, C % 4]`. The outputs are input cells, so the
//! circuit only exposes them.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let a = inputs.floats("a", &[6, 3, 4])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let a = q.load_floats(ctx, &a);
    let mut grid = Vec::with_capacity(72);
    for row in 0..6 {
        for col in 0..12 {
            let block = (row / 3) * 3 + col / 4;
            grid.push(a[block * 12 + (row % 3) * 4 + col % 4]);
        }
    }
    out.floats(&grid);
    Ok(())
}
