//! DS-1000 case 0343, halo2-base baseline.
//! Kernel: `[value for value in X.T.flat]` on `Integer[5, 6]`: the 30 elements in
//! column-major order, `X[i, j]` at position `j * 5 + i`.

use crate::prelude::*;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let x = inputs.ints("X", &[5, 6])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let x = q.load_ints(ctx, &x);
    let transposed: Vec<Cell> = (0..6).flat_map(|j| (0..5).map(move |i| (i, j))).map(|(i, j)| x[i * 6 + j]).collect();
    out.ints(&transposed);
    Ok(())
}
