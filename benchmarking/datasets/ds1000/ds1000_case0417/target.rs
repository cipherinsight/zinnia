//! DS-1000 case 0417, halo2-base baseline.
//! Kernel: `data[::-1][:(8 // 4) * 4].reshape(-1, 4).mean(axis=1)` with the
//! constant `bin_size = 4` on `Float[8]`: the means of the reversed data's
//! bins, `data[7:3:-1]` and `data[3::-1]`.
//! A Q32 mean of 4 values is `floor(sum / 4)` (the quotient
//! `floor(sum * 2^32 / (4 * 2^32))`), a division by the constant 4: a 2-bit
//! remainder check and a window bound on the quotient. The reversal is a
//! fixed relabelling of the cells.

use crate::prelude::*;

const BIN_SIZE: usize = 4;

pub fn circuit(builder: &mut Builder, inputs: &Inputs, out: &mut Outputs) -> Result<()> {
    let data = inputs.floats("data", &[8])?;
    let q = Q32::new(builder);
    let ctx = builder.main(0);

    let data = q.load_floats(ctx, &data);
    let reversed: Vec<Cell> = data.iter().rev().copied().collect();
    let means: Vec<Cell> = reversed
        .chunks(BIN_SIZE)
        .map(|bin| {
            let total = q.sum(ctx, bin.iter().copied());
            q.floor_div_const(ctx, total, BIN_SIZE as u64).0.cell
        })
        .collect();
    out.floats(&means);
    Ok(())
}
