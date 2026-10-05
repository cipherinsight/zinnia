//! DS-1000 case 0415, halo2-base baseline.
//! Kernel: `data[:(8 // 4) * 4].reshape(-1, 4).max(axis=1)` with the constant
//! `bin_size = 4` on `Float[8]`: the maxima of `data[0:4]` and `data[4:8]`.
//! Each maximum is `q.max_of`: a hint pinned by four range checks and a product
//! (instead of four window bounds and a three-comparison chain).

use crate::prelude::*;

const BIN_SIZE: usize = 4;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_floats(ctx, &data);
    let maxima: Vec<Cell> = data.chunks(BIN_SIZE).map(|bin| q.max_of(ctx, bin)).collect();
    out.floats(&maxima);
    Ok(())
}
